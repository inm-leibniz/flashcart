import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import tiny_model_cfg, write_tiny_extxyz  # noqa: E402

from flashcart.data.dataset import make_loader  # noqa: E402
from flashcart.model.flashcart import FlashCartPotential  # noqa: E402

pytestmark = pytest.mark.slow

_N_EPOCHS = 2


def _worker(fabric, train_path: Path, out_dir: Path) -> None:
    from flashcart.training.loss import LossAccumulator, PropertyLossFunction
    from flashcart.utils.distributed import is_global_zero, per_rank_batch_size

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    cfg = tiny_model_cfg()
    model = FlashCartPotential.from_config(cfg)

    if is_global_zero(fabric):
        prefit_loader = make_loader(
            train_path,
            cfg["elements"],
            model.r_max,
            batch_size=2,
            energy_key="REF_energy",
            forces_key="REF_forces",
            shuffle=False,
            dynamic_batching=False,
        )
        model.pre_fit(prefit_loader)
    if fabric.world_size > 1:
        fabric.barrier()
        model.broadcast_pre_fit_state(fabric)

    per_rank_batch = per_rank_batch_size(2, fabric.world_size)
    train_loader = make_loader(
        train_path,
        cfg["elements"],
        model.r_max,
        batch_size=per_rank_batch,
        energy_key="REF_energy",
        forces_key="REF_forces",
        shuffle=False,
        dynamic_batching=False,
        n_replicas=fabric.world_size,
        rank=fabric.global_rank,
        balance_batches=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    model, optimizer = fabric.setup(model, optimizer)
    if hasattr(model, "mark_forward_method"):
        model.mark_forward_method("predict")
    train_loader = fabric.setup_dataloaders(train_loader, use_distributed_sampler=False)

    train_loss = PropertyLossFunction("energy", reduce="sse", normalization="none")
    eval_loss = PropertyLossFunction("energy", reduce="mae", normalization="none")

    model.train()
    losses: list[float] = []
    accumulator = LossAccumulator(eval_loss)
    for _ in range(_N_EPOCHS):
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            results = model.predict(batch, compute_forces=False)
            loss = train_loss.training_loss(results, batch, world_size=fabric.world_size)
            fabric.backward(loss)
            optimizer.step()
            losses.append(loss.detach().item())
            accumulator.update(results, batch)
    train_metric = accumulator.compute(fabric).item()

    params = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    payload = {
        "world_size": fabric.world_size,
        "rank": fabric.global_rank,
        "params": params.tolist(),
        "losses": losses,
        "train_metric": train_metric,
    }
    out_file = out_dir / f"result_rank{fabric.global_rank}_of{fabric.world_size}.json"
    out_file.write_text(json.dumps(payload))


def _run_worker(tmp_path: Path, world_size: int) -> dict[int, dict]:
    train_path = tmp_path / "train.extxyz"
    write_tiny_extxyz(train_path)
    out_dir = tmp_path / f"out_w{world_size}"
    out_dir.mkdir(exist_ok=True)

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, __file__, str(train_path), str(out_dir), str(world_size)],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
        cwd=repo_root,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    payloads = {}
    for rank in range(world_size):
        out_file = out_dir / f"result_rank{rank}_of{world_size}.json"
        assert out_file.exists(), f"rank {rank} wrote no result\nstderr:\n{result.stderr}"
        payloads[rank] = json.loads(out_file.read_text())
    return payloads


@pytest.fixture(scope="module")
def ddp_results(tmp_path_factory) -> dict[int, dict[int, dict]]:
    base = tmp_path_factory.mktemp("fabric_ddp")
    return {w: _run_worker(base, w) for w in (1, 2)}


def test_ddp_ranks_end_with_identical_parameters(ddp_results) -> None:
    rank0 = torch.tensor(ddp_results[2][0]["params"])
    rank1 = torch.tensor(ddp_results[2][1]["params"])
    torch.testing.assert_close(rank0, rank1, atol=0.0, rtol=0.0)


def test_ddp_training_matches_single_process_parameters(ddp_results) -> None:
    single = torch.tensor(ddp_results[1][0]["params"])
    ddp = torch.tensor(ddp_results[2][0]["params"])
    torch.testing.assert_close(ddp, single, atol=1.0e-9, rtol=1.0e-9)
    assert all(torch.isfinite(torch.tensor(p["losses"])).all() for p in ddp_results[2].values())
    assert torch.isfinite(torch.tensor(ddp_results[2][0]["train_metric"]))


def test_pre_fit_state_tensors_cover_all_broadcast_state(tmp_path) -> None:
    train_path = tmp_path / "train.extxyz"
    write_tiny_extxyz(train_path)

    cfg = tiny_model_cfg()
    rank0 = FlashCartPotential.from_config(cfg)
    rank1 = FlashCartPotential.from_config(cfg)

    loader = make_loader(
        train_path,
        cfg["elements"],
        rank0.r_max,
        batch_size=2,
        energy_key="REF_energy",
        forces_key="REF_forces",
        shuffle=False,
        dynamic_batching=False,
    )
    rank0.pre_fit(loader)

    for dst, src in zip(rank1._pre_fit_state_tensors(), rank0._pre_fit_state_tensors()):
        dst.copy_(src)
    rank1.avg_neighbors = rank0.avg_neighbors

    torch.testing.assert_close(rank1.scale_shift.shifts, rank0.scale_shift.shifts)
    torch.testing.assert_close(rank1.scale_shift.scales, rank0.scale_shift.scales)
    for dst_layer, src_layer in zip(rank1.interactions, rank0.interactions):
        torch.testing.assert_close(dst_layer.sqrt_avg_neighbors, src_layer.sqrt_avg_neighbors)
    assert rank0.scale_shift.shifts.abs().sum().item() > 0.0


if __name__ == "__main__":
    from lightning.fabric import Fabric

    if len(sys.argv) != 4:
        raise SystemExit(f"Usage: {Path(__file__).name} TRAIN.extxyz OUT_DIR WORLD_SIZE")
    _train, _out, _world = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    _fabric = Fabric(
        accelerator="cpu",
        devices=_world,
        strategy="ddp_spawn" if _world > 1 else "auto",
        precision="64-true",
    )
    _fabric.launch(_worker, _train, _out)
