import pytest
import torch

from helpers import batch_graphs, tiny_graph, tiny_potential, write_tiny_extxyz
from flashcart.data.dataset import make_loader
from flashcart.data.padding import PadAtomicData, slice_padded_outputs
from flashcart.training.tasks import make_predict_padder, padded_predict, seed_padder_from_loader

pytestmark = pytest.mark.usefixtures("fp64_default")

R_MAX = 3.0


def _graphs(n_graphs: int, dtype=torch.float64) -> list:
    torch.manual_seed(1234)
    graphs = []
    for i in range(n_graphs):
        n = 4 + i
        positions = torch.randn(n, 3, dtype=dtype) * 1.2
        atom_types = torch.randint(0, 2, (n,), dtype=torch.long)
        graphs.append(tiny_graph(positions, atom_types, R_MAX))
    return graphs


def _predict_with_grads(model, batch, padder):
    results = padded_predict(
        model,
        batch,
        padder,
        predict_kwargs={"compute_forces": True},
        compile_kwargs={},
        create_graph=True,
    )
    loss = results["energy"].square().sum() + results["forces"].square().sum()
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    return (
        {k: v.detach().clone() for k, v in results.items() if torch.is_tensor(v)},
        grads,
    )


@pytest.mark.parametrize("n_graphs", [1, 3])
def test_padded_predict_matches_unpadded(n_graphs: int) -> None:
    model = tiny_potential().to(dtype=torch.float64)
    batch = batch_graphs(_graphs(n_graphs))
    padder = PadAtomicData(r_max=R_MAX, atom_multiple=8, edge_multiple=16, graph_multiple=2)

    padded_out, padded_grads = _predict_with_grads(model, batch, padder)
    plain_out, plain_grads = _predict_with_grads(model, batch_graphs(_graphs(n_graphs)), None)

    torch.testing.assert_close(padded_out["energy"], plain_out["energy"], atol=1.0e-10, rtol=1.0e-10)
    torch.testing.assert_close(padded_out["forces"], plain_out["forces"], atol=1.0e-10, rtol=1.0e-10)
    assert padded_grads.keys() == plain_grads.keys()
    for name in plain_grads:
        torch.testing.assert_close(padded_grads[name], plain_grads[name], atol=1.0e-10, rtol=1.0e-10)


def test_predictions_independent_of_padding_budget() -> None:
    model = tiny_potential().to(dtype=torch.float64)
    small = PadAtomicData(r_max=R_MAX, atom_multiple=8, edge_multiple=16, graph_multiple=2)
    large = PadAtomicData(r_max=R_MAX, atom_multiple=64, edge_multiple=128, graph_multiple=8)

    out_small, _ = _predict_with_grads(model, batch_graphs(_graphs(2)), small)
    out_large, _ = _predict_with_grads(model, batch_graphs(_graphs(2)), large)

    assert large.total_atoms > small.total_atoms
    torch.testing.assert_close(out_small["energy"], out_large["energy"], atol=1.0e-10, rtol=1.0e-10)
    torch.testing.assert_close(out_small["forces"], out_large["forces"], atol=1.0e-10, rtol=1.0e-10)


def test_padder_reuses_buffers_until_budget_grows() -> None:
    padder = PadAtomicData(r_max=R_MAX, atom_multiple=8, edge_multiple=16, graph_multiple=2)

    padded, _, _ = padder(batch_graphs(_graphs(2)))
    first_positions = padded.positions

    padded, _, _ = padder(batch_graphs(_graphs(2)))
    assert padded.positions is first_positions

    padded, _, _ = padder(batch_graphs(_graphs(6)))
    assert padded.positions is not first_positions
    assert padder.total_atoms > 0 and padded.positions.shape[0] == padder.total_atoms


def test_backward_after_repad_raises() -> None:
    model = tiny_potential().to(dtype=torch.float64)
    padder = PadAtomicData(r_max=R_MAX, atom_multiple=8, edge_multiple=16, graph_multiple=2)

    results = padded_predict(
        model,
        batch_graphs(_graphs(2)),
        padder,
        predict_kwargs={"compute_forces": True},
        compile_kwargs={},
        create_graph=True,
    )
    loss = results["energy"].square().sum() + results["forces"].square().sum()

    padder(batch_graphs(_graphs(2)))

    with pytest.raises(RuntimeError):
        loss.backward()
    model.zero_grad(set_to_none=True)


def test_seeded_budgets_match_sampler_maxima(tmp_path) -> None:
    train_path = tmp_path / "train.extxyz"
    write_tiny_extxyz(train_path, n_structures=6)
    loader = make_loader(
        train_path,
        ["H"],
        R_MAX,
        batch_size=4,
        energy_key="REF_energy",
        forces_key="REF_forces",
        shuffle=False,
        dynamic_batching=True,
    )
    sampler = loader.batch_sampler

    padder = make_predict_padder(
        tiny_potential(elements=["H"], atomic_shifts=[0.0]), predict_compile=True, pad_kwargs={}
    )
    seed_padder_from_loader(padder, loader)
    assert (padder.atom_budget, padder.edge_budget, padder.graph_budget) == (
        sampler.max_nodes,
        sampler.max_edges,
        sampler.max_graphs,
    )

    padder.seed_budgets(atom_budget=1, edge_budget=1, graph_budget=1)
    assert (padder.atom_budget, padder.edge_budget, padder.graph_budget) == (
        sampler.max_nodes,
        sampler.max_edges,
        sampler.max_graphs,
    )
