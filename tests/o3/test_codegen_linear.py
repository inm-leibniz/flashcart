from types import SimpleNamespace

import pytest

from helpers import exec_with_triton_stubs
from flashcart.o3._codegen_linear import build_triton_linear_module_source


def _exec_generated_module(kernel_l_max: int) -> dict:
    return exec_with_triton_stubs(build_triton_linear_module_source(kernel_l_max))


@pytest.mark.parametrize("kernel_l_max", [0, 1, 3, 4, 7])
def test_emitted_module_matches_expected_api(kernel_l_max: int) -> None:
    num_slots = kernel_l_max + 1
    src = build_triton_linear_module_source(kernel_l_max)
    ns = _exec_generated_module(kernel_l_max)

    assert ns["KERNEL_L_MAX"] == kernel_l_max
    assert ns["NUM_L_SLOTS"] == num_slots
    assert callable(ns["linear_rowwise_kernel"])
    assert callable(ns["linear_wgrad_kernel"])

    for i in range(num_slots):
        for name in (f"w{i}_ptr", f"v{i}_ptr", f"g{i}_ptr", f"K{i}", f"O{i}", f"FIN{i}", f"FOUT{i}", f"M{i}"):
            assert name in src, name
    for name in (f"w{num_slots}_ptr", f"g{num_slots}_ptr", f"K{num_slots}", f"FIN{num_slots}"):
        assert name not in src, name


def test_emitted_fp64_config_prune() -> None:
    ns = _exec_generated_module(1)
    prune = ns["_prune_fp64_tiles"]
    configs = [
        SimpleNamespace(kwargs={"BLOCK_ROWS": 128, "BLOCK_OUT": 128, "BLOCK_K": 64}),
        SimpleNamespace(kwargs={"BLOCK_ROWS": 64, "BLOCK_OUT": 128, "BLOCK_K": 32}),
        SimpleNamespace(kwargs={"BLOCK_ROWS": 32, "BLOCK_OUT": 32, "BLOCK_K": 32}),
        SimpleNamespace(kwargs={"BLOCK_ROWS": 64, "BLOCK_OUT": 64, "BLOCK_K": 32}),
    ]
    kept = prune(configs, {"USE_FP64": True})
    assert all(c.kwargs["BLOCK_ROWS"] * c.kwargs["BLOCK_OUT"] <= 64 * 64 for c in kept)
    assert len(kept) == 2
    assert prune(configs, {"USE_FP64": False}) == configs
    big_only = configs[:1]
    assert prune(big_only, {"USE_FP64": True}) == big_only


def test_emitted_smem_config_prune() -> None:
    ns = _exec_generated_module(1)
    ns["_smem_limit_bytes"] = lambda: 101376
    prune = ns["_prune_smem_exceeding"]

    rowwise = [
        SimpleNamespace(kwargs={"BLOCK_ROWS": 128, "BLOCK_OUT": 128, "BLOCK_K": 64}, num_stages=3),
        SimpleNamespace(kwargs={"BLOCK_ROWS": 64, "BLOCK_OUT": 128, "BLOCK_K": 32}, num_stages=3),
        SimpleNamespace(kwargs={"BLOCK_ROWS": 32, "BLOCK_OUT": 32, "BLOCK_K": 32}, num_stages=3),
    ]
    kept = prune(rowwise, {}, TERM1=True, TERM2=False, USE_FP64=False)
    assert kept == rowwise[1:]
    kept = prune(rowwise, {}, TERM1=True, TERM2=True, USE_FP64=False)
    assert kept == rowwise[1:]
    big_only = rowwise[:1]
    assert prune(big_only, {}, TERM1=True, TERM2=False, USE_FP64=False) == big_only

    wgrad = [
        SimpleNamespace(kwargs={"BLOCK_FOUT": 64, "BLOCK_FIN": 64, "BLOCK_K": 64}, num_stages=3),
        SimpleNamespace(kwargs={"BLOCK_FOUT": 32, "BLOCK_FIN": 32, "BLOCK_K": 32}, num_stages=3),
    ]
    assert prune(wgrad, {}, USE_FP64=False) == wgrad
    assert prune(wgrad, {}, USE_FP64=True) == wgrad[1:]

    composed = ns["_prune_rowwise_configs"]
    kept = composed(rowwise, {}, TERM1=True, TERM2=False, USE_FP64=True)
    assert kept == [rowwise[2]]
