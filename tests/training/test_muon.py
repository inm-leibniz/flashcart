import copy

import pytest

pytest.importorskip("torch.optim._muon", reason="torch.optim.Muon requires torch >= 2.9")

import torch

from helpers import tiny_potential
from flashcart.training.muon import NS_COEFF_FAST, NS_COEFF_POLISH, HybridMuon, _batched_newtonschulz
from flashcart.training.optimizers import make_optimizer

SHAPES = [(16, 8), (16, 8), (8, 16), (1, 32)]
KWARGS = dict(lr=1e-2, weight_decay=1e-3, adjust_lr_fn="match_rms_adamw")


def _make_params(seed: int = 0) -> list[torch.nn.Parameter]:
    torch.manual_seed(seed)
    return [torch.nn.Parameter(torch.randn(s)) for s in SHAPES]


def _set_grads(params, seed: int) -> None:
    torch.manual_seed(seed)
    for p in params:
        p.grad = torch.randn_like(p)


def _run(opt, params, n_steps, seed0=100):
    start = [p.detach().clone() for p in params]
    for step in range(n_steps):
        _set_grads(params, seed0 + step)
        opt.step()
    return [p.detach() - s for p, s in zip(params, start)]


def _assert_deltas_close(deltas_a, deltas_b, cos_min=0.999, rel_max=0.01):
    for da, db in zip(deltas_a, deltas_b):
        cos = torch.nn.functional.cosine_similarity(da.reshape(1, -1), db.reshape(1, -1)).item()
        rel = ((da - db).norm() / da.norm().clamp(min=1e-12)).item()
        assert cos > cos_min and rel < rel_max, f"shape {tuple(da.shape)}: cos={cos:.5f}, rel={rel:.4f}"


def test_hybrid_matches_vanilla_muon_damping_off() -> None:
    pa, pb = _make_params(), _make_params()
    da = _run(torch.optim.Muon(pa, **KWARGS), pa, 10)
    db = _run(HybridMuon(pb, alignment_damping=False, **KWARGS), pb, 10)
    _assert_deltas_close(da, db)


def test_hybrid_loads_vanilla_muon_state_dict() -> None:
    params = _make_params()
    vanilla = torch.optim.Muon(params, **KWARGS)
    _set_grads(params, 1)
    vanilla.step()
    bufs = [vanilla.state[p]["momentum_buffer"].clone() for p in params]

    hybrid = HybridMuon(params, alignment_damping=True, **KWARGS)
    hybrid.load_state_dict(copy.deepcopy(vanilla.state_dict()))
    for p, buf in zip(params, bufs):
        torch.testing.assert_close(hybrid.state[p]["momentum_buffer"], buf)
    _set_grads(params, 2)
    hybrid.step()
    assert all(torch.isfinite(p).all() for p in params)


def test_interrupted_run_matches_uninterrupted_run() -> None:
    pa, pb = _make_params(), _make_params()
    opt_a = HybridMuon(pa, alignment_damping=True, **KWARGS)
    _run(opt_a, pa, 5)

    opt_b1 = HybridMuon(pb, alignment_damping=True, **KWARGS)
    _run(opt_b1, pb, 3)
    saved = copy.deepcopy(opt_b1.state_dict())
    opt_b2 = HybridMuon(pb, alignment_damping=True, **KWARGS)
    opt_b2.load_state_dict(saved)
    _run(opt_b2, pb, 2, seed0=103)

    for qa, qb in zip(pa, pb):
        torch.testing.assert_close(qa, qb)


def test_step_ignores_params_with_missing_grads() -> None:
    params = _make_params()
    opt = HybridMuon(params, alignment_damping=True, **KWARGS)
    frozen = params[3].detach().clone()
    for step in range(3):
        _set_grads(params, 300 + step)
        params[3].grad = None
        opt.step()
    torch.testing.assert_close(params[3].detach(), frozen)
    _set_grads(params, 310)
    opt.step()
    assert not torch.equal(params[3].detach(), frozen)
    assert all(torch.isfinite(p).all() for p in params)


def test_damping_slows_gradients_opposing_established_momentum(monkeypatch) -> None:
    params = [torch.nn.Parameter(torch.randn(8, 8))]
    opt = HybridMuon(params, alignment_damping=True, lr=1e-2, weight_decay=0.0, adjust_lr_fn="match_rms_adamw")
    torch.manual_seed(0)
    base = torch.randn_like(params[0])
    scales = []
    orig = HybridMuon._batched_magma_scale

    def record(self, grads_stack, bufs_stack, bucket):
        scale = orig(self, grads_stack, bufs_stack, bucket)
        scales.append(float(scale.max()))
        return scale

    monkeypatch.setattr(HybridMuon, "_batched_magma_scale", record)
    for _ in range(30):
        params[0].grad = base.clone()
        opt.step()
    coherent_scale = scales[-1]
    for _ in range(12):
        params[0].grad = -base
        opt.step()

    assert coherent_scale > 0.9
    assert min(scales[30:]) < 0.45
    assert all(0.1 <= s <= 1.0 for s in scales)


def test_make_optimizer_routes_model_parameters() -> None:
    model = tiny_potential()
    main_opt, aux = make_optimizer(model, optimizer="muon", lr=1e-2, weight_decay=1e-3)

    muon_ids = {id(p) for group in aux[0].param_groups for p in group["params"]}
    main_ids = {id(p) for group in main_opt.param_groups for p in group["params"]}
    all_ids = {id(p) for p in model.parameters() if p.requires_grad}
    assert muon_ids | main_ids == all_ids
    assert muon_ids & main_ids == set()
    assert {id(p) for p in model.non_muon_parameters()} <= main_ids
    assert all(p.ndim == 2 for group in aux[0].param_groups for p in group["params"])

    assert id(model.readout.readout_mlp[-1].weight) in main_ids
    assert all(min(p.shape) > 1 for group in aux[0].param_groups for p in group["params"])
    assert id(model.readout.readout_mlp[0].weight) in muon_ids

    names = aux[0].param_groups[0]["param_names"]
    assert len(names) == len(aux[0].param_groups[0]["params"])
    assert all(isinstance(n, str) for n in names)

    main_only, aux_none = make_optimizer(model, optimizer="adamw", lr=1e-2, weight_decay=1e-3)
    assert aux_none is None
    adamw_ids = {id(p) for group in main_only.param_groups for p in group["params"]}
    assert adamw_ids == all_ids
