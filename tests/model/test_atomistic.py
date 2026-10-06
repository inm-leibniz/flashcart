from pathlib import Path

import pytest
import torch

from helpers import TRITON_TEST_AVAILABLE, tiny_graph, tiny_potential
from flashcart.model.flashcart import FlashCartPotential
from flashcart.o3 import _tensor_product as tensor_product

R_MAX = 3.0

_DEVICES = [
    pytest.param("cpu", False, id="cpu"),
    pytest.param(
        "cuda",
        True,
        id="triton",
        marks=pytest.mark.skipif(not TRITON_TEST_AVAILABLE, reason="Triton requires CUDA."),
    ),
]


def _random_inputs(dtype: torch.dtype, device: torch.device, n_atoms: int = 6) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.randn(n_atoms, 3, dtype=dtype, device=device) * 1.5
    atom_types = torch.randint(0, 2, (n_atoms,), dtype=torch.long, device=device)
    return positions, atom_types


def test_from_checkpoint_matches_original_predictions(tmp_path: Path):
    torch.manual_seed(1234)
    model = tiny_potential()
    positions, atom_types = _random_inputs(torch.get_default_dtype(), torch.device("cpu"))
    graph = tiny_graph(positions, atom_types, R_MAX)

    ckpt = tmp_path / "best"
    model.save_inference_checkpoint(ckpt, meta={"epoch": 1})
    loaded = FlashCartPotential.from_checkpoint(ckpt, map_location="cpu")

    expected = model.predict(graph, compute_forces=True)
    actual = loaded.predict(graph, compute_forces=True)
    torch.testing.assert_close(actual["energy"], expected["energy"], atol=0.0, rtol=0.0)
    torch.testing.assert_close(actual["forces"], expected["forces"], atol=0.0, rtol=0.0)


def test_from_checkpoint_raises_on_missing_directory(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        FlashCartPotential.from_checkpoint(tmp_path / "missing")


def test_training_step_updates_parameters():
    torch.manual_seed(1234)
    dtype = torch.float64
    model = tiny_potential().to(dtype=dtype)
    graph = tiny_graph(*_random_inputs(dtype, torch.device("cpu")), R_MAX)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-3)
    before = [p.detach().clone() for p in model.parameters()]

    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        out = model.predict(graph, compute_forces=True, create_graph=True)
        loss = out["energy"].square().sum() + out["forces"].square().sum()
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()

    changed = any(not torch.equal(p, b) for p, b in zip(model.parameters(), before))
    assert changed, "two SGD steps left every parameter untouched"


@pytest.mark.slow
@pytest.mark.parametrize(("device_name", "use_triton"), _DEVICES)
def test_compiled_predict_matches_eager(device_name: str, use_triton: bool) -> None:
    torch.manual_seed(1234)
    dtype = torch.float64
    tol = 1.0e-10
    device = torch.device(device_name)
    model = tiny_potential(use_triton=use_triton).to(device=device, dtype=dtype)
    graph = tiny_graph(*_random_inputs(dtype, device), R_MAX)

    scatter_cache = getattr(tensor_product, "_SCATTER_DUMMY", None)
    if scatter_cache is not None:
        scatter_cache.clear()
    torch._dynamo.reset()

    compiled = model.predict(graph, compute_forces=True, use_compile=True)
    if scatter_cache is not None:
        assert all(
            type(v) is torch.Tensor for v in scatter_cache.values()
        ), "a tensor created while tracing leaked into the scatter placeholder cache"

    eager = model.predict(graph, compute_forces=True)
    for key in ("energy", "forces"):
        torch.testing.assert_close(compiled[key], eager[key], atol=tol, rtol=tol)

    optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-3)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        out = model.predict(graph, compute_forces=True, create_graph=True, use_compile=True)
        loss = out["energy"].square().sum() + out["forces"].square().sum()
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
