import numpy as np
import pytest
import torch

from helpers import TRITON_TEST_AVAILABLE, random_rotation, tiny_graph, tiny_periodic_graph, tiny_potential
from flashcart.model.flashcart import FlashCartPotential

L_MAX_EDGE_ATTRS = 2
R_MAX = 5.0
HESSIAN_PARAM_NAME = "readout.readout_mlp.2.weight"


def _model(use_triton: bool, device: torch.device, dtype: torch.dtype) -> FlashCartPotential:
    return (
        FlashCartPotential(
            elements=["H", "C", "N", "O"],
            r_max=R_MAX,
            n_hidden_feats=4,
            l_max_hidden_feats=2,
            l_max_edge_attrs=L_MAX_EDGE_ATTRS,
            n_radial=4,
            n_interactions=2,
            correlation=3,
            nonlinearity=True,
            layer_norm=True,
            hidden_radial=[8],
            hidden_readout=[8],
            fit_atomic_shifts=False,
            atomic_shifts=[0.0, 0.0, 0.0, 0.0],
            use_triton=use_triton,
        )
        .to(device=device, dtype=dtype)
        .eval()
    )


def _random_inputs(dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.randn(6, 3, dtype=dtype, device=device) * 1.5
    atom_types = torch.randint(0, 4, (6,), dtype=torch.long, device=device)
    return positions, atom_types


def test_predict_rotation_equivariant() -> None:
    torch.manual_seed(1234)
    dtype = torch.float64
    device = torch.device("cpu")
    model = _model(False, device, dtype)
    positions, atom_types = _random_inputs(dtype, device)
    R = random_rotation(seed=1234, dtype=dtype, device=device)

    out = model.predict(tiny_graph(positions, atom_types, R_MAX), compute_forces=True)
    out_rot = model.predict(tiny_graph(positions @ R.T, atom_types, R_MAX), compute_forces=True)

    expected_forces = out["forces"] @ R.T
    assert torch.allclose(out_rot["energy"], out["energy"], atol=1.0e-9, rtol=1.0e-9)
    assert torch.allclose(out_rot["forces"], expected_forces, atol=1.0e-9, rtol=1.0e-9)


def test_predict_translation_invariant() -> None:
    torch.manual_seed(1234)
    dtype = torch.float64
    model = _model(False, torch.device("cpu"), dtype)
    positions, atom_types = _random_inputs(dtype, torch.device("cpu"))
    shift = torch.tensor([1.7, -0.3, 2.9], dtype=dtype)

    out = model.predict(tiny_graph(positions, atom_types, R_MAX), compute_forces=True)
    out_shifted = model.predict(tiny_graph(positions + shift, atom_types, R_MAX), compute_forces=True)

    assert torch.allclose(out_shifted["energy"], out["energy"], atol=1.0e-10, rtol=1.0e-10)
    assert torch.allclose(out_shifted["forces"], out["forces"], atol=1.0e-10, rtol=1.0e-10)


def test_predict_permutation_equivariant() -> None:
    torch.manual_seed(1234)
    dtype = torch.float64
    model = _model(False, torch.device("cpu"), dtype)
    positions, atom_types = _random_inputs(dtype, torch.device("cpu"))
    perm = torch.randperm(positions.shape[0])

    out = model.predict(tiny_graph(positions, atom_types, R_MAX), compute_forces=True)
    out_perm = model.predict(tiny_graph(positions[perm], atom_types[perm], R_MAX), compute_forces=True)

    assert torch.allclose(out_perm["energy"], out["energy"], atol=1.0e-10, rtol=1.0e-10)
    assert torch.allclose(out_perm["forces"], out["forces"][perm], atol=1.0e-10, rtol=1.0e-10)


def _batch_two(graph_a, graph_b, dtype: torch.dtype):
    from flashcart.utils.torch_geometric.data import Data

    n_a = graph_a.positions.shape[0]
    n_b = graph_b.positions.shape[0]
    e_a = graph_a.edge_index.shape[1]
    e_b = graph_b.edge_index.shape[1]
    return Data(
        positions=torch.cat([graph_a.positions, graph_b.positions]),
        atom_types=torch.cat([graph_a.atom_types, graph_b.atom_types]),
        edge_index=torch.cat([graph_a.edge_index, graph_b.edge_index + n_a], dim=1),
        batch=torch.cat([torch.zeros(n_a, dtype=torch.long), torch.ones(n_b, dtype=torch.long)]),
        n_atoms=torch.tensor([n_a, n_b], dtype=torch.long),
        cell=torch.zeros(2, 3, 3, dtype=dtype),
        shifts=torch.zeros(e_a + e_b, 3, dtype=dtype),
        pbc=torch.tensor([[False, False, False], [False, False, False]]),
        use_shifts=False,
    )


def test_batched_predict_matches_per_graph() -> None:
    torch.manual_seed(1234)
    dtype = torch.float64
    model = _model(False, torch.device("cpu"), dtype)
    pos_a, types_a = _random_inputs(dtype, torch.device("cpu"))
    torch.manual_seed(4321)
    pos_b = torch.randn(4, 3, dtype=dtype) * 1.5
    types_b = torch.randint(0, 4, (4,), dtype=torch.long)

    graph_a = tiny_graph(pos_a, types_a, R_MAX)
    graph_b = tiny_graph(pos_b, types_b, R_MAX)
    out_a = model.predict(graph_a, compute_forces=True)
    out_b = model.predict(graph_b, compute_forces=True)
    out = model.predict(_batch_two(graph_a, graph_b, dtype), compute_forces=True)

    assert torch.allclose(out["energy"], torch.cat([out_a["energy"], out_b["energy"]]), atol=1.0e-10, rtol=1.0e-10)
    assert torch.allclose(out["forces"], torch.cat([out_a["forces"], out_b["forces"]]), atol=1.0e-10, rtol=1.0e-10)


@pytest.mark.usefixtures("fp64_default")
def test_stress_matches_finite_difference() -> None:
    torch.manual_seed(1234)
    dtype = torch.float64
    model = tiny_potential().to(dtype=dtype).eval()
    a = 3.5
    graph = tiny_periodic_graph(
        atomic_numbers=np.array([1, 8]),
        positions=np.array([[0.05, 0.10, 0.0], [1.75, 0.0, 0.1]]),
        cell=a * np.eye(3),
        elements=["H", "O"],
        r_max=3.0,
    )
    assert bool((graph.shifts != 0).any()), "geometry must produce periodic image edges"

    out = model.predict(graph, compute_stress=True)
    volume = torch.det(graph.cell.squeeze(0).to(dtype))

    def energy_at(strain: torch.Tensor) -> torch.Tensor:
        from flashcart.utils.torch_geometric.data import Data

        transform = torch.eye(3, dtype=dtype) + strain
        fields = dict(graph.to_dict())
        fields["positions"] = graph.positions.to(dtype) @ transform.T
        fields["cell"] = graph.cell.to(dtype) @ transform.T
        return model.predict(Data(**fields))["energy"].squeeze(0)

    h = 1.0e-5
    fd = torch.zeros(3, 3, dtype=dtype)
    for i in range(3):
        for j in range(3):
            strain = torch.zeros(3, 3, dtype=dtype)
            strain[i, j] = h
            e_plus = energy_at(strain)
            strain[i, j] = -h
            e_minus = energy_at(strain)
            fd[i, j] = (e_plus - e_minus) / (2.0 * h)
    fd = (fd + fd.T) / (2.0 * volume)

    from flashcart.utils.geometry import to_voigt6

    torch.testing.assert_close(out["stress"].squeeze(0), to_voigt6(fd), atol=1.0e-7, rtol=1.0e-6)


def test_gradcheck_py() -> None:
    torch.manual_seed(1234)
    dtype = torch.float64
    device = torch.device("cpu")
    model = _model(False, device, dtype)
    positions, atom_types = _random_inputs(dtype, device)
    positions = positions.detach().requires_grad_(True)
    names = [name for name, _ in model.named_parameters()]

    def fn(pos: torch.Tensor, *ws: torch.Tensor) -> torch.Tensor:
        graph = tiny_graph(pos, atom_types, R_MAX).to_dict()
        return torch.func.functional_call(model, dict(zip(names, ws)), (graph,)).sum()

    params = tuple(p.detach().clone().requires_grad_(True) for p in model.parameters())
    assert torch.autograd.gradcheck(fn, (positions, *params), fast_mode=True)

    params = tuple(
        p.detach().clone().requires_grad_(name == HESSIAN_PARAM_NAME) for name, p in model.named_parameters()
    )
    assert torch.autograd.gradgradcheck(fn, (positions, *params), fast_mode=True)


def _observables(model: FlashCartPotential, positions: torch.Tensor, atom_types: torch.Tensor, cotangents) -> dict:
    positions = positions.detach().clone().requires_grad_(True)
    energy = model(tiny_graph(positions, atom_types, R_MAX).to_dict()).sum()
    named = dict(model.named_parameters())
    wrt = [positions, *named.values()]

    first = torch.autograd.grad(energy, wrt, create_graph=True, retain_graph=True, allow_unused=True)
    first_filled = [torch.zeros_like(w) if g is None else g for g, w in zip(first, wrt)]

    hvps = []
    for cot in cotangents:
        scalar = sum((g * c).sum() for g, c in zip(first_filled, cot) if g.requires_grad)
        hvp = torch.autograd.grad(scalar, wrt, retain_graph=True, allow_unused=True)
        hvps.append(
            torch.cat([torch.zeros_like(w).reshape(-1) if h is None else h.reshape(-1) for h, w in zip(hvp, wrt)])
        )

    return {
        "energy": energy.detach().reshape(1),
        "forces": (-first_filled[0]).detach(),
        "param_grads": {name: g.detach() for name, g in zip(named, first_filled[1:])},
        "hvps": torch.stack(hvps).detach(),
    }


@pytest.mark.skipif(not TRITON_TEST_AVAILABLE, reason="Triton requires CUDA.")
def test_triton_matches_py() -> None:
    torch.manual_seed(1234)
    dtype = torch.float64
    device = torch.device("cuda")
    tol = 1.0e-12
    reference = _model(False, device, dtype)
    candidate = _model(True, device, dtype)
    candidate.load_state_dict(reference.state_dict())
    positions, atom_types = _random_inputs(dtype, device)

    generator = torch.Generator(device="cpu").manual_seed(99)
    shapes = [positions.shape, *(p.shape for p in reference.parameters())]
    cotangents = [[torch.randn(s, dtype=dtype, generator=generator).to(device) for s in shapes] for _ in range(3)]

    expected = _observables(reference, positions, atom_types, cotangents)
    actual = _observables(candidate, positions, atom_types, cotangents)

    for key in ("energy", "forces", "hvps"):
        diff = (actual[key] - expected[key]).abs().max()
        assert torch.allclose(
            actual[key], expected[key], atol=tol, rtol=tol
        ), f"{key} differs. Max diff: {diff.item():.6e}"

    assert actual["param_grads"].keys() == expected["param_grads"].keys()
    for name, grad in actual["param_grads"].items():
        diff = (grad - expected["param_grads"][name]).abs().max()
        assert torch.allclose(
            grad, expected["param_grads"][name], atol=tol, rtol=tol
        ), f"Gradient for {name} differs. Max diff: {diff.item():.6e}"
