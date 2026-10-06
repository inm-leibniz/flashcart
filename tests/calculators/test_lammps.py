from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import torch
from helpers import requires_cuda

from flashcart.calculators.lammps_mliap import (
    FlashCartLAMMPSUnified,
    build_lammps_unified,
    graph_from_mliap,
    save_lammps_unified,
)
from flashcart.model.flashcart import FlashCartPotential
from flashcart.utils.torch_geometric.data import Data


class _FakeMLIAPData:
    def __init__(
        self,
        positions: np.ndarray,
        cell: Optional[np.ndarray] = None,
        nlocal: Optional[int] = None,
        device: str = "cpu",
    ):
        n = len(positions)
        self.ntotal = n
        self.nlocal = n if nlocal is None else nlocal
        self.nghosts = self.ntotal - self.nlocal
        self.npairs = 0
        self.device = torch.device(device)
        self.elems = self._array(np.zeros(n, dtype=np.int32))
        self.pair_i = self._array(np.array([], dtype=np.int32))
        self.pair_j = self._array(np.array([], dtype=np.int32))
        self.rij = self._array(np.zeros((0, 3), dtype=np.float64))
        self._positions = positions
        self._cell = cell
        self.forward_exchange_calls = 0
        self.reverse_exchange_calls = 0
        self.ghost_owner = {i: min(i, self.nlocal - 1) for i in range(self.nlocal, self.ntotal) if self.nlocal > 0}

    def _array(self, values: np.ndarray):
        return torch.as_tensor(values, device=self.device) if self.device.type == "cuda" else values

    def _build_pairs(self, cutoff: float):
        pairs_i, pairs_j, rijs = [], [], []
        for i in range(self.ntotal):
            for j in range(self.ntotal):
                if i == j:
                    continue
                d = self._positions[j] - self._positions[i]
                if np.linalg.norm(d) < cutoff:
                    pairs_i.append(i)
                    pairs_j.append(j)
                    rijs.append(d)
        self.set_pairs(pairs_i, pairs_j, np.array(rijs, dtype=np.float64).reshape(-1, 3))

    def set_pairs(self, pair_i, pair_j, rij):
        self.pair_i = self._array(np.asarray(pair_i, dtype=np.int32))
        self.pair_j = self._array(np.asarray(pair_j, dtype=np.int32))
        self.rij = self._array(np.asarray(rij, dtype=np.float64))
        self.npairs = len(pair_i)

    def forward_exchange(self, feats, out, vec_len: int):
        self.forward_exchange_calls += 1
        out.copy_(feats)
        for ghost, owner in self.ghost_owner.items():
            out[ghost].copy_(feats[owner])

    def reverse_exchange(self, grad, gout, vec_len: int):
        self.reverse_exchange_calls += 1
        gout.copy_(grad)
        for ghost, owner in self.ghost_owner.items():
            gout[owner].add_(grad[ghost])
            gout[ghost].zero_()


class _NativeMLIAPData(_FakeMLIAPData):
    def __getattribute__(self, name):
        if name in ("forward_exchange", "reverse_exchange"):
            raise AttributeError(name)
        return super().__getattribute__(name)


def _model(n_interactions: int = 1, r_max: float = 3.0, use_triton: bool = False) -> FlashCartPotential:
    return FlashCartPotential(
        elements=["H", "O"],
        r_max=r_max,
        n_hidden_feats=4,
        l_max_hidden_feats=1,
        l_max_edge_attrs=2,
        n_radial=4,
        n_interactions=n_interactions,
        correlation=2,
        nonlinearity=False,
        layer_norm=False,
        hidden_radial=[8],
        hidden_readout=[8],
        fit_atomic_shifts=False,
        atomic_shifts=[0.0, 0.0],
        use_triton=use_triton,
    )


_DEVICES = ["cpu", pytest.param("cuda", marks=requires_cuda)]


def _numpy(values):
    return values.cpu().numpy() if torch.is_tensor(values) else values


def test_graph_from_mliap_separates_local_and_ghost_atoms():
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [1.8, 0.0, 0.0],
        ]
    )
    fake = _FakeMLIAPData(pos, nlocal=2)
    fake._build_pairs(cutoff=2.0)

    graph = graph_from_mliap(fake, ["H"], r_max=2.0, device=torch.device("cpu"))
    assert isinstance(graph, Data)
    assert graph.num_nodes == 3
    assert graph.nlocal == 2
    assert graph.ntotal == 3
    assert graph.nghost == 1
    assert graph.n_atoms.tolist() == [2]
    assert graph.batch.shape == (2,)
    assert graph.atom_types.shape == (3,)
    assert graph.edge_index.shape[0] == 2
    assert graph.vectors.shape == (graph.edge_index.shape[1], 3)


def test_reloaded_lammps_unified_matches_original(tmp_path: Path):
    model = _model()
    ckpt = tmp_path / "best"
    model.save_inference_checkpoint(ckpt)

    unified = build_lammps_unified(ckpt)
    assert isinstance(unified, FlashCartLAMMPSUnified)
    assert unified.device == torch.device("cpu")
    assert unified.element_types == model.elements
    assert unified.rcutfac == 0.5 * unified.r_max

    out = tmp_path / "flashcart-mliap.pt"
    save_lammps_unified(unified, out)
    loaded = torch.load(out, weights_only=False)
    assert isinstance(loaded, FlashCartLAMMPSUnified)
    assert loaded.element_types == model.elements


_BOX = 1.8
_X1 = 0.85


def _periodic_chain_serial() -> _FakeMLIAPData:
    fake = _FakeMLIAPData(np.array([[0.0, 0.0, 0.0], [_X1, 0.0, 0.0]]))
    right = _X1
    left = _X1 - _BOX
    fake.set_pairs(
        pair_i=[0, 0, 1, 1],
        pair_j=[1, 1, 0, 0],
        rij=[[right, 0.0, 0.0], [left, 0.0, 0.0], [-right, 0.0, 0.0], [-left, 0.0, 0.0]],
    )
    fake.f = np.zeros((2, 3), dtype=np.float64)
    fake.eatoms = np.zeros(2, dtype=np.float64)
    return fake


def _periodic_chain_decomposed() -> _FakeMLIAPData:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [_X1, 0.0, 0.0],
            [_BOX, 0.0, 0.0],
            [_X1 - _BOX, 0.0, 0.0],
        ]
    )
    fake = _FakeMLIAPData(positions, nlocal=2)
    fake.ghost_owner = {2: 0, 3: 1}
    fake._build_pairs(cutoff=1.0)
    fake.f = np.zeros((4, 3), dtype=np.float64)
    fake.eatoms = np.zeros(2, dtype=np.float64)
    return fake


@pytest.mark.parametrize("n_interactions", [1, 2, 3])
def test_mliap_forces_match_undecomposed(n_interactions: int):
    torch.manual_seed(7)
    unified = FlashCartLAMMPSUnified(_model(n_interactions=n_interactions, r_max=1.0).double())

    serial = _periodic_chain_serial()
    unified.compute_forces(serial)

    decomposed = _periodic_chain_decomposed()
    unified.compute_forces(decomposed)
    assert decomposed.forward_exchange_calls == n_interactions - 1
    assert decomposed.reverse_exchange_calls == n_interactions - 1

    folded = decomposed.f[:2].copy()
    for ghost, owner in decomposed.ghost_owner.items():
        folded[owner] += decomposed.f[ghost]

    np.testing.assert_allclose(decomposed.eatoms, serial.eatoms, atol=1.0e-10, rtol=1.0e-10)
    np.testing.assert_allclose(folded, serial.f[:2], atol=1.0e-10, rtol=1.0e-10)
    assert np.abs(serial.f[:2]).max() > 0.0


def _random_decomposed(seed: int, nlocal: int, nghost: int, cutoff: float, device: str = "cpu") -> _FakeMLIAPData:
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0.0, 1.5, size=(nlocal + nghost, 3))
    fake = _FakeMLIAPData(positions, nlocal=nlocal, device=device)
    fake.ghost_owner = {nlocal + k: int(rng.integers(nlocal)) for k in range(nghost)}
    fake._build_pairs(cutoff=cutoff)
    fake.elems = fake._array((np.arange(nlocal + nghost) % 2).astype(np.int32))
    fake.f = fake._array(np.zeros((nlocal + nghost, 3), dtype=np.float64))
    fake.eatoms = fake._array(np.zeros(nlocal, dtype=np.float64))
    return fake


@pytest.mark.slow
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("compile_mode", ["default", "max-autotune-no-cudagraphs"])
def test_compiled_unified_matches_eager_and_reuses_graph_across_shapes(compile_mode: str, device: str):
    torch.manual_seed(11)
    model = _model(n_interactions=2, r_max=1.0, use_triton=device == "cuda").double()
    eager = FlashCartLAMMPSUnified(model)
    compiled = FlashCartLAMMPSUnified(model, compile_mode=compile_mode)
    assert compiled.compile_mode == compile_mode

    torch._dynamo.reset()
    counters = torch._dynamo.utils.counters
    counters.clear()
    for seed, nlocal, nghost in ((0, 5, 4), (1, 6, 7), (2, 4, 9)):
        fake_eager = _random_decomposed(seed, nlocal, nghost, cutoff=1.0, device=device)
        fake_compiled = _random_decomposed(seed, nlocal, nghost, cutoff=1.0, device=device)
        assert fake_eager.npairs > 3
        eager.compute_forces(fake_eager)
        compiled.compute_forces(fake_compiled)
        assert fake_compiled.forward_exchange_calls == 1
        assert fake_compiled.reverse_exchange_calls == 1
        np.testing.assert_allclose(_numpy(fake_compiled.eatoms), _numpy(fake_eager.eatoms), atol=1.0e-9, rtol=1.0e-9)
        np.testing.assert_allclose(_numpy(fake_compiled.f), _numpy(fake_eager.f), atol=1.0e-9, rtol=1.0e-9)
        assert fake_compiled.energy == pytest.approx(fake_eager.energy, abs=1.0e-9)
    assert counters["stats"]["unique_graphs"] == 1


def test_rank_without_ghosts_still_exchanges():
    torch.manual_seed(3)
    unified = FlashCartLAMMPSUnified(_model(n_interactions=2, r_max=1.0).double())
    serial = _periodic_chain_serial()
    unified.compute_forces(serial)
    assert serial.forward_exchange_calls == 1 and serial.reverse_exchange_calls == 1
    assert np.isfinite(serial.f).all() and np.abs(serial.f).max() > 0.0


def test_ranks_without_atoms_or_pairs_raise():
    unified = FlashCartLAMMPSUnified(_model(n_interactions=2, r_max=1.0).double())
    empty = _FakeMLIAPData(np.zeros((0, 3)), nlocal=0)
    with pytest.raises(RuntimeError, match="neighbor pairs"):
        unified.compute_forces(empty)
    lonely = _FakeMLIAPData(np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]))
    with pytest.raises(RuntimeError, match="neighbor pairs"):
        unified.compute_forces(lonely)


def test_attached_lammps_exits_on_unsupported_rank(monkeypatch, capsys):
    unified = FlashCartLAMMPSUnified(_model(n_interactions=2).double())
    unified.interface = object()

    def exit_process(status):
        raise SystemExit(status)

    monkeypatch.setattr("flashcart.calculators.lammps_mliap.os._exit", exit_process)
    with pytest.raises(SystemExit) as exc:
        unified.compute_forces(_FakeMLIAPData(np.zeros((0, 3))))
    assert exc.value.code == 1
    assert "owns 0 atoms with 0 neighbor pairs" in capsys.readouterr().err


@pytest.mark.parametrize("n_interactions", [1, 2])
def test_native_build_without_exchange_hooks(n_interactions):
    torch.manual_seed(2)
    unified = FlashCartLAMMPSUnified(_model(n_interactions=n_interactions, r_max=1.0).double())
    reference = _periodic_chain_serial()
    unified.compute_forces(reference)

    native = _NativeMLIAPData(np.array([[0.0, 0.0, 0.0], [_X1, 0.0, 0.0]]))
    native.set_pairs(reference.pair_i, reference.pair_j, reference.rij)
    native.f = np.zeros((2, 3))
    native.eatoms = np.zeros(2)
    unified.compute_forces(native)
    np.testing.assert_allclose(native.eatoms, reference.eatoms, atol=1.0e-12, rtol=1.0e-12)
    np.testing.assert_allclose(native.f, reference.f, atol=1.0e-12, rtol=1.0e-12)

    reference = _periodic_chain_decomposed()
    unified.compute_forces(reference)
    with_ghosts = _periodic_chain_decomposed()
    with_ghosts.__class__ = _NativeMLIAPData
    if n_interactions == 1:
        unified.compute_forces(with_ghosts)
        np.testing.assert_allclose(with_ghosts.eatoms, reference.eatoms, atol=1.0e-12, rtol=1.0e-12)
        np.testing.assert_allclose(with_ghosts.f, reference.f, atol=1.0e-12, rtol=1.0e-12)
    else:
        with pytest.raises(RuntimeError, match="KOKKOS"):
            unified.compute_forces(with_ghosts)


@pytest.mark.slow
def test_compiled_exchange_accepts_noncontiguous_inputs():
    from flashcart.utils.compile import configure_autograd_for_compile
    from flashcart.utils.lammps import lammps_data_slot

    configure_autograd_for_compile(allow_autograd=True)
    fake = _FakeMLIAPData(np.zeros((5, 3)), nlocal=3)
    fake.ghost_owner = {3: 0, 4: 2}
    feats = torch.randn(4, 5, dtype=torch.float64).t().requires_grad_(True)
    assert not feats.is_contiguous()

    def exchange_and_grad(x):
        out = torch.ops.flashcart.lammps_forward_exchange(x)
        weight = torch.arange(20, dtype=x.dtype).view(4, 5).t()
        (grad,) = torch.autograd.grad((out * weight).sum(), x)
        return out.detach(), grad

    torch._dynamo.reset()
    with lammps_data_slot(fake):
        out_eager, grad_eager = exchange_and_grad(feats)
        out_compiled, grad_compiled = torch.compile(exchange_and_grad, fullgraph=True, dynamic=True)(feats)
    assert out_compiled.is_contiguous() and grad_compiled.is_contiguous()
    torch.testing.assert_close(out_compiled, out_eager)
    torch.testing.assert_close(grad_compiled, grad_eager)
    assert fake.forward_exchange_calls == 2 and fake.reverse_exchange_calls == 2


@pytest.mark.slow
def test_compiled_reverse_exchange_accepts_noncontiguous_inputs():
    from flashcart.utils.lammps import lammps_data_slot

    fake = _FakeMLIAPData(np.zeros((5, 3)), nlocal=3)
    fake.ghost_owner = {3: 0, 4: 2}
    grads = torch.arange(20, dtype=torch.float64).view(4, 5).t()
    assert not grads.is_contiguous()

    def reverse_exchange(x):
        return torch.ops.flashcart.lammps_reverse_exchange(x) * 2

    torch._dynamo.reset()
    with lammps_data_slot(fake):
        expected = reverse_exchange(grads)
        actual = torch.compile(reverse_exchange, fullgraph=True)(grads)

    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected)
    assert fake.reverse_exchange_calls == 2


def test_compile_settings_survive_export_without_cached_functions(tmp_path: Path):
    with pytest.raises(ValueError):
        FlashCartLAMMPSUnified(_model(), compile_mode="reduce-overhead")
    unified = FlashCartLAMMPSUnified(
        _model(),
        compile_mode="default",
        compile_fullgraph=False,
    )

    unified._predict_fn()
    assert unified.model._compiled_predict_cache

    out = save_lammps_unified(unified, tmp_path / "flashcart-mliap.pt")
    loaded = torch.load(out, weights_only=False)

    assert loaded.compile_mode == "default"
    assert loaded.compile_fullgraph is False
    assert loaded.model._compiled_predict_cache == {}


def test_compute_forces_skips_energy_copy_when_eflag_false():
    unified = FlashCartLAMMPSUnified(_model())

    pos = np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    fake = _FakeMLIAPData(pos)
    fake._build_pairs(cutoff=2.0)
    fake.elems = np.array([0, 0], dtype=np.int32)
    fake.f = np.zeros((2, 3), dtype=np.float64)
    fake.eatoms = np.zeros(2, dtype=np.float64)
    fake.eflag = 0

    unified.compute_forces(fake)
    assert np.isfinite(fake.f).all()
    assert not hasattr(fake, "energy")
