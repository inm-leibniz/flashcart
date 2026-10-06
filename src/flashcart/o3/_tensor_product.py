"""Evaluate Cartesian tensor products with generated PyTorch and Triton code.

Generated modules use the maximum tensor rank read at import. The Triton
backend registers forward and derivative operations with PyTorch. Python functions
in this module select kernel launches for independent rows or neighbor aggregation
and group output ranks and derivative outputs when needed. PyTorch evaluation generates
additional expressions when the requested ranks exceed the cached limit.
"""

import os
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from flashcart.o3._codegen_cache import load_generated
from flashcart.o3._codegen_tensor_product import (
    build_py_tensor_product_forward_module_source,
    build_triton_tp_module_source,
    compile_py_tensor_product_forward,
)
from flashcart.o3.kernel_config import kernel_l_max
from flashcart.utils.env import env_bool

KERNEL_L_MAX: int = kernel_l_max()

_GENERATED_MODULE_NAME = "flashcart.o3._tensor_product_kernels"
_CODEGEN_FILE = Path(__file__).with_name("_codegen_tensor_product.py")
_PY_FORWARD_MODULE_NAME = "flashcart.o3._py_tensor_product_forward"


_PY_TENSOR_PRODUCT_FORWARD_MODULE = load_generated(
    _PY_FORWARD_MODULE_NAME, _CODEGEN_FILE, KERNEL_L_MAX, build_py_tensor_product_forward_module_source
)


try:
    import triton
    import triton.language as tl  # noqa: F401
    from torch.library import triton_op

    from flashcart.o3._triton_launch import launcher

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


def _slice_offsets(slices, l_max: int, kernel_l_max: int) -> List[int]:
    """Pad the per-l start offsets out to ``kernel_l_max + 1`` entries."""
    return [int(slices[l][0]) if l <= l_max else 0 for l in range(kernel_l_max + 1)]


def _pad_path_counts(n_paths, kernel_l_max: int) -> List[int]:
    return [int(n_paths[l]) if l < len(n_paths) else 0 for l in range(kernel_l_max + 1)]


def _out_block_sizes(
    n_paths: List[int],
    out_features: int,
    reduce_paths: bool,
) -> List[int]:
    sizes = []
    for l, n in enumerate(n_paths):
        dim_l = 2 * l + 1
        if reduce_paths:
            sizes.append(dim_l * out_features if n > 0 else 0)
        else:
            sizes.append(n * dim_l * out_features)
    return sizes


@dataclass
class TPTuning:
    """Collect tensor-product launch settings read from the environment.

    The module-level instance is created at import. Changing environment variables
    afterward does not update it. Group budgets count output components per feature
    channel. The remaining fields select how neighbors are processed, how derivative
    calculations are split, and how many edge groups are assigned to a receiver.
    """

    csr_group_budget: int = 100
    bwd_group_budget: int = 100
    dbwd_group_budget: Optional[int] = None
    bwd_csr_min_deg: int = 64
    bwd_csr_edges_per_slot: int = 32
    bwd_csr_max_slots: int = 64
    bwd_csr_force_slots: Optional[int] = None
    scatter_kernel: str = "csr"
    bwd_split_override: Optional[str] = None
    dbwd_split_override: Optional[str] = None
    peak_aware_groups: bool = True

    @classmethod
    def from_env(cls) -> "TPTuning":
        """Read tensor-product tuning settings from ``FLASHCART_TP_*`` variables.

        Returns:
            TPTuning: Settings using the declared defaults for unset variables.
        """
        def opt_int(name: str) -> Optional[int]:
            return int(os.environ[name]) if name in os.environ else None

        return cls(
            peak_aware_groups=env_bool("FLASHCART_TP_PEAK_AWARE_GROUPS", True),
            csr_group_budget=int(os.environ.get("FLASHCART_TP_CSR_GROUP_BUDGET", "100")),
            bwd_group_budget=int(os.environ.get("FLASHCART_TP_BWD_GROUP_BUDGET", "100")),
            dbwd_group_budget=opt_int("FLASHCART_TP_DBWD_GROUP_BUDGET"),
            bwd_csr_min_deg=int(os.environ.get("FLASHCART_TP_BWD_CSR_MIN_DEG", "64")),
            bwd_csr_force_slots=opt_int("FLASHCART_TP_BWD_CSR_SLOTS"),
            scatter_kernel=os.environ.get("FLASHCART_TP_SCATTER_KERNEL", "csr"),
            bwd_split_override=os.environ.get("FLASHCART_TP_BWD_SPLIT"),
            dbwd_split_override=os.environ.get("FLASHCART_TP_DBWD_SPLIT"),
        )


TUNING = TPTuning.from_env()


def _bwd_csr_chunk_slots(n_batch: int, out_batch: int) -> int:
    """Choose how many edge groups share a receiver's backward calculation.

    Args:
        n_batch (int): Number of edges.
        out_batch (int): Number of receiver rows.

    Returns:
        int: Explicit slot override when set, clamped to at least one.
            Otherwise, choose from the average edge count, threshold, and
            configured slot limits.
    """
    if TUNING.bwd_csr_force_slots is not None:
        return max(1, TUNING.bwd_csr_force_slots)
    avg_deg = n_batch // max(out_batch, 1)
    if avg_deg < TUNING.bwd_csr_min_deg:
        return 1
    step = TUNING.bwd_csr_edges_per_slot
    return min(TUNING.bwd_csr_max_slots, (avg_deg + step - 1) // step)


_SPLIT_PLANS = {
    "bwd": {
        "fused": ((1, 1, 1),),
        "full": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    },
    "dbwd": {
        "fused": ((1, 1, 1, 1),),
        "full": ((1, 0, 0, 0), (0, 1, 1, 0), (0, 0, 0, 1)),
    },
}


_SPLIT_FULL_GATE = {"bwd": 5, "dbwd_csr": 5, "dbwd_edge": 4}


def _resolve_split_plan(kind: str, out_l_max: int) -> Tuple[Tuple[int, ...], ...]:
    """Choose which derivative outputs share a kernel launch.

    Args:
        kind (str): ``"bwd"``, ``"dbwd_csr"``, or ``"dbwd_edge"``.
        out_l_max (int): Maximum output tensor rank.

    Returns:
        tuple[tuple[int, ...], ...]: One output-selection mask per launch.
            Backward masks select the two input gradients and weight gradient.
            Double-backward masks select derivatives with respect to the
            incoming gradient, first input, second input, and weights.
    """
    family = "bwd" if kind == "bwd" else "dbwd"
    override = TUNING.bwd_split_override if kind == "bwd" else TUNING.dbwd_split_override
    name = override if override is not None else ("full" if out_l_max >= _SPLIT_FULL_GATE[kind] else "fused")
    plans = _SPLIT_PLANS[family]
    return plans.get(name, plans["fused"])


def _csr_path_groups(
    out_l_max: int,
    n_paths_per_l: List[int],
    reduce_paths: bool,
    budget: Optional[int] = None,
    peak_aware: bool = True,
) -> List[int]:
    """Assign consecutive output tensor ranks to kernel launch groups.

    The budget limits stored output components per feature channel, rather than bytes or
    measured register use. A tensor rank is never split across groups. Peak-aware
    grouping raises the budget to fit the largest active rank.

    Args:
        out_l_max (int): Highest output tensor rank to group.
        n_paths_per_l (list[int]): Path count for each output tensor rank.
            Ranks without paths add no components to a group.
        reduce_paths (bool): Count one component block per rank when paths
            are summed, or one block per path otherwise.
        budget (int, optional): Component budget per feature channel.
            Default: None, which uses the forward grouping budget.
        peak_aware (bool, optional): Raise the budget to fit the largest
            active rank when also enabled by the environment setting.
            Default: True.

    Returns:
        list[int]: Group index for each provided output tensor rank.
    """
    if budget is None:
        budget = TUNING.csr_group_budget
    peak_aware = peak_aware and TUNING.peak_aware_groups
    last_l = min(out_l_max, len(n_paths_per_l) - 1)
    comps_l = [(2 * l + 1) * (1 if reduce_paths else n_paths_per_l[l]) for l in range(last_l + 1)]
    if peak_aware:
        budget = max(
            budget,
            max((c for l, c in enumerate(comps_l) if n_paths_per_l[l]), default=0),
        )
    groups = [0] * len(n_paths_per_l)
    group, used = 0, 0
    for l in range(last_l + 1):
        if n_paths_per_l[l] == 0:
            groups[l] = group
            continue
        comps = comps_l[l]
        if used > 0 and used + comps > budget:
            group += 1
            used = 0
        groups[l] = group
        used += comps
    return groups


def _launch_path_groups(autotuner, launch_one, n_groups: int, zero_on_retune=()) -> None:
    """Launch each path group and repeat if tuning resets shared outputs.

    Autotuning a later group can clear contributions from earlier groups. If tuning adds
    a cache entry, multiple groups were launched, and shared outputs were provided,
    clear those outputs and repeat all groups with the selected configurations.
    """
    n_tuned = len(autotuner.cache)
    for path_group in range(n_groups):
        launch_one(path_group)
    if n_groups > 1 and zero_on_retune and len(autotuner.cache) != n_tuned:
        for t in zero_on_retune:
            if t.numel():
                t.zero_()
        for path_group in range(n_groups):
            launch_one(path_group)


def _run_with_retune_guard(autotuner, run_all, zero_tensors, multi: bool) -> None:
    """Repeat a sequence of launches if tuning resets accumulated outputs.

    When ``multi`` is true and tuning adds a cache entry, clear the provided outputs and
    rerun the sequence with the selected configurations.
    """
    n_tuned = len(autotuner.cache)
    run_all()
    if multi and len(autotuner.cache) != n_tuned:
        for t in zero_tensors:
            if t.numel():
                t.zero_()
        run_all()


if torch.cuda.is_available() and TRITON_AVAILABLE:

    _kernels = load_generated(_GENERATED_MODULE_NAME, _CODEGEN_FILE, KERNEL_L_MAX, build_triton_tp_module_source)

    tp_fwd_kernel = _kernels.tp_fwd_kernel
    tp_fwd_csr_kernel = _kernels.tp_fwd_csr_kernel
    tp_bwd_kernel = _kernels.tp_bwd_kernel
    tp_bwd_csr_kernel = _kernels.tp_bwd_csr_kernel
    tp_dbwd_kernel = _kernels.tp_dbwd_kernel
    tp_dbwd_csr_kernel = _kernels.tp_dbwd_csr_kernel

    _GEN_L_MAX = int(_kernels.KERNEL_L_MAX)
    _NUM_L_SLOTS = _GEN_L_MAX + 1

    _SCATTER_DUMMY: dict = {}

    def _resolve_scatter(idx_i, idx_j, in1, in2=None):
        """Resolve node and edge dimensions for a tensor-product launch.

        Args:
            idx_i (torch.Tensor or None): Receiver indices.
            idx_j (torch.Tensor or None): Sender indices.
            in1 (torch.Tensor): First input, with one row per node in convolution mode.
            in2 (torch.Tensor, optional): Second input, with one row per edge in
                convolution mode. Default: None, which uses the receiver-index count.

        Returns:
            tuple[bool, torch.Tensor, torch.Tensor, int, int]: Whether convolution
                is enabled, contiguous receiver and sender indices, the work-row
                count, and the output-row count. Convolution processes edge rows
                but retains one output row per first-input node. Independent-row
                calls receive dummy index tensors.
        """
        in1_batch = in1.shape[0]
        needs_scatter = idx_i is not None and idx_j is not None
        if needs_scatter:
            idx_i_c = idx_i.contiguous()
            idx_j_c = idx_j.contiguous()
            n_batch = idx_i_c.shape[0] if in2 is None else in2.shape[0]
            out_batch = in1_batch
            return True, idx_i_c, idx_j_c, n_batch, out_batch
        if in2 is not None and in1_batch != in2.shape[0]:
            raise ValueError(
                "idx_i and idx_j must be provided when in1 and in2 have different "
                f"batch sizes. Got in1_batch={in1_batch}, in2_batch={in2.shape[0]}."
            )
        dummy = _SCATTER_DUMMY.get(in1.device)
        if dummy is None:
            dummy = in1.new_zeros(1, dtype=torch.int64)
            if type(dummy) is torch.Tensor:
                _SCATTER_DUMMY[in1.device] = dummy
        return False, dummy, dummy, in1_batch, in1_batch

    def _alloc_out_sizes(
        reduce_paths: bool,
        out_features: int,
        n_paths_per_l: List[int],
        out_sizes: List[int],
    ) -> List[int]:
        if reduce_paths:
            return [(2 * l + 1) * out_features if n_paths_per_l[l] > 0 else 0 for l in range(len(n_paths_per_l))]
        return list(out_sizes)

    def _per_l_kwargs(prefix: str, values: List[int], suffix: str = "") -> dict:
        """Expand a per-l list into ``{prefix}{l}{suffix}=values[l]`` kwargs."""
        return {f"{prefix}{l}{suffix}": int(values[l]) for l in range(_NUM_L_SLOTS)}

    @triton_op("flashcart::tp_fwd", mutates_args={})
    def tp_fwd_op(
        in1: torch.Tensor,
        in2: torch.Tensor,
        weights: torch.Tensor,
        idx_i: Optional[torch.Tensor],
        idx_j: Optional[torch.Tensor],
        in1_l_max: int,
        in2_l_max: int,
        out_l_max: int,
        in1_features: int,
        in2_features: int,
        symmetric_product: bool,
        shared_weights: bool,
        in1_offs: List[int],
        in2_offs: List[int],
        out_sizes: List[int],
        n_paths_per_l: List[int],
        n_total_paths: int,
        reduce_paths: bool = False,
    ) -> torch.Tensor:
        in1 = in1.contiguous()
        in2 = in2.contiguous()
        weights = weights.contiguous()
        in1_batch = in1.shape[0]
        needs_scatter, idx_i_ptr, idx_j_ptr, n_batch, out_batch = _resolve_scatter(idx_i, idx_j, in1, in2)
        out_features = max(in1_features, in2_features)
        allocs = _alloc_out_sizes(reduce_paths, out_features, n_paths_per_l, out_sizes)

        if needs_scatter and TUNING.scatter_kernel == "edge":
            out = torch.zeros((out_batch, sum(allocs)), device=in1.device, dtype=in1.dtype)
            edge_grid = lambda meta: (
                triton.cdiv(n_batch, meta["BLOCK_SIZE"]),
                triton.cdiv(out_features, meta["FEATURE_BLOCK"]),
            )
            launcher(tp_fwd_kernel)[edge_grid](
                in1,
                in2,
                out,
                weights,
                idx_i_ptr,
                idx_j_ptr,
                n_batch=n_batch,
                IN1_FEATURES=in1_features,
                IN2_FEATURES=in2_features,
                OUT_FEATURES=out_features,
                in1_stride=in1.stride(0),
                in2_stride=in2.stride(0),
                out_stride=out.stride(0),
                IN1_L_MAX=in1_l_max,
                IN2_L_MAX=in2_l_max,
                OUT_L_MAX=out_l_max,
                SYMMETRIC_PRODUCT=int(symmetric_product),
                SHARED_WEIGHTS=int(shared_weights),
                NEEDS_SCATTER=1,
                **_per_l_kwargs("IN1_OFF_L", in1_offs),
                **_per_l_kwargs("IN2_OFF_L", in2_offs),
                **_per_l_kwargs("OUT_L", allocs, suffix="_SIZE"),
                **_per_l_kwargs("N_PATHS_L", n_paths_per_l),
                N_TOTAL_PATHS=n_total_paths,
                REDUCE_PATHS=int(reduce_paths),
            )
            return out

        if needs_scatter:
            torch._assert_async(
                (idx_i_ptr[1:] >= idx_i_ptr[:-1]).all(),
                "Scatter tensor product requires a receiver-sorted idx_i "
                "(edges grouped by ascending destination index).",
            )
            rowptr_i = torch.searchsorted(
                idx_i_ptr,
                torch.arange(out_batch + 1, device=idx_i_ptr.device, dtype=idx_i_ptr.dtype),
            )
            out = torch.empty((out_batch, sum(allocs)), device=in1.device, dtype=in1.dtype)
            csr_grid = lambda meta: (
                out_batch,
                triton.cdiv(out_features, meta["FEATURE_BLOCK"]),
            )
            groups_l = _csr_path_groups(out_l_max, n_paths_per_l, reduce_paths)
            n_groups = max(groups_l[: out_l_max + 1]) + 1

            def _launch(path_group: int) -> None:
                launcher(tp_fwd_csr_kernel)[csr_grid](
                    in1,
                    in2,
                    out,
                    weights,
                    idx_j_ptr,
                    rowptr_i,
                    in1.stride(0),
                    in2.stride(0),
                    out.stride(0),
                    IN1_FEATURES=in1_features,
                    IN2_FEATURES=in2_features,
                    OUT_FEATURES=out_features,
                    IN1_L_MAX=in1_l_max,
                    IN2_L_MAX=in2_l_max,
                    OUT_L_MAX=out_l_max,
                    SYMMETRIC_PRODUCT=int(symmetric_product),
                    SHARED_WEIGHTS=int(shared_weights),
                    **_per_l_kwargs("IN1_OFF_L", in1_offs),
                    **_per_l_kwargs("IN2_OFF_L", in2_offs),
                    **_per_l_kwargs("OUT_L", allocs, suffix="_SIZE"),
                    **_per_l_kwargs("N_PATHS_L", n_paths_per_l),
                    N_TOTAL_PATHS=n_total_paths,
                    REDUCE_PATHS=int(reduce_paths),
                    PATH_GROUP=path_group,
                    **_per_l_kwargs("GROUP_L", groups_l),
                )

            _launch_path_groups(tp_fwd_csr_kernel, _launch, n_groups)
            return out

        alloc = torch.zeros if reduce_paths else torch.empty
        out = alloc((out_batch, sum(allocs)), device=in1.device, dtype=in1.dtype)
        grid = lambda meta: (
            triton.cdiv(n_batch, meta["BLOCK_SIZE"]),
            triton.cdiv(out_features, meta["FEATURE_BLOCK"]),
        )
        launcher(tp_fwd_kernel)[grid](
            in1,
            in2,
            out,
            weights,
            idx_i_ptr,
            idx_j_ptr,
            n_batch=n_batch,
            IN1_FEATURES=in1_features,
            IN2_FEATURES=in2_features,
            OUT_FEATURES=out_features,
            in1_stride=in1.stride(0),
            in2_stride=in2.stride(0),
            out_stride=out.stride(0),
            IN1_L_MAX=in1_l_max,
            IN2_L_MAX=in2_l_max,
            OUT_L_MAX=out_l_max,
            SYMMETRIC_PRODUCT=int(symmetric_product),
            SHARED_WEIGHTS=int(shared_weights),
            NEEDS_SCATTER=0,
            **_per_l_kwargs("IN1_OFF_L", in1_offs),
            **_per_l_kwargs("IN2_OFF_L", in2_offs),
            **_per_l_kwargs("OUT_L", allocs, suffix="_SIZE"),
            **_per_l_kwargs("N_PATHS_L", n_paths_per_l),
            N_TOTAL_PATHS=n_total_paths,
            REDUCE_PATHS=int(reduce_paths),
        )
        return out

    def _save_tp_scalars_to_ctx(
        ctx,
        in1_l_max,
        in2_l_max,
        out_l_max,
        in1_features,
        in2_features,
        symmetric_product,
        shared_weights,
        in1_offs,
        in2_offs,
        out_sizes,
        n_paths_per_l,
        n_total_paths,
        reduce_paths,
    ):
        ctx.in1_l_max = in1_l_max
        ctx.in2_l_max = in2_l_max
        ctx.out_l_max = out_l_max
        ctx.in1_features = in1_features
        ctx.in2_features = in2_features
        ctx.symmetric_product = symmetric_product
        ctx.shared_weights = shared_weights
        ctx.in1_offs = list(in1_offs)
        ctx.in2_offs = list(in2_offs)
        ctx.out_sizes = list(out_sizes)
        ctx.n_paths_per_l = list(n_paths_per_l)
        ctx.n_total_paths = n_total_paths
        ctx.reduce_paths = reduce_paths

    def tp_setup_context(ctx, inputs, output):
        (
            in1,
            in2,
            weights,
            idx_i,
            idx_j,
            in1_l_max,
            in2_l_max,
            out_l_max,
            in1_features,
            in2_features,
            symmetric_product,
            shared_weights,
            in1_offs,
            in2_offs,
            out_sizes,
            n_paths_per_l,
            n_total_paths,
            reduce_paths,
        ) = inputs
        ctx.save_for_backward(in1, in2, weights, idx_i, idx_j)
        _save_tp_scalars_to_ctx(
            ctx,
            in1_l_max,
            in2_l_max,
            out_l_max,
            in1_features,
            in2_features,
            symmetric_product,
            shared_weights,
            in1_offs,
            in2_offs,
            out_sizes,
            n_paths_per_l,
            n_total_paths,
            reduce_paths,
        )

    def tp_bwd(ctx, grad_out: torch.Tensor):
        need_grad_in1 = bool(ctx.needs_input_grad[0])
        need_grad_in2 = bool(ctx.needs_input_grad[1])
        need_grad_weights = bool(ctx.needs_input_grad[2])
        if not (need_grad_in1 or need_grad_in2 or need_grad_weights):
            return (None, None, None) + (None,) * (len(ctx.needs_input_grad) - 3)

        in1, in2, weights, idx_i, idx_j = ctx.saved_tensors
        grad_in1, grad_in2, grad_weights = torch.ops.flashcart.tp_bwd(
            grad_out,
            in1.contiguous(),
            in2.contiguous(),
            weights.contiguous(),
            idx_i,
            idx_j,
            ctx.in1_l_max,
            ctx.in2_l_max,
            ctx.out_l_max,
            ctx.in1_features,
            ctx.in2_features,
            ctx.symmetric_product,
            ctx.shared_weights,
            ctx.in1_offs,
            ctx.in2_offs,
            ctx.out_sizes,
            ctx.n_paths_per_l,
            ctx.n_total_paths,
            ctx.reduce_paths,
            need_grad_in1,
            need_grad_in2,
            need_grad_weights,
        )
        return (
            grad_in1 if need_grad_in1 else None,
            grad_in2 if need_grad_in2 else None,
            grad_weights if need_grad_weights else None,
        ) + (None,) * (len(ctx.needs_input_grad) - 3)

    @triton_op("flashcart::tp_bwd", mutates_args={})
    def tp_bwd_op(
        grad_out: torch.Tensor,
        in1: torch.Tensor,
        in2: torch.Tensor,
        weights: torch.Tensor,
        idx_i: Optional[torch.Tensor],
        idx_j: Optional[torch.Tensor],
        in1_l_max: int,
        in2_l_max: int,
        out_l_max: int,
        in1_features: int,
        in2_features: int,
        symmetric_product: bool,
        shared_weights: bool,
        in1_offs: List[int],
        in2_offs: List[int],
        out_sizes: List[int],
        n_paths_per_l: List[int],
        n_total_paths: int,
        reduce_paths: bool,
        need_grad_in1: bool,
        need_grad_in2: bool,
        need_grad_weights: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        in1 = in1.contiguous()
        in2 = in2.contiguous()
        weights = weights.contiguous()
        out_features = max(in1_features, in2_features)
        needs_scatter, idx_i_s, idx_j_s, n_batch, out_batch = _resolve_scatter(idx_i, idx_j, in1)

        _zero_g1 = needs_scatter or in1_features == 1
        _zero_g2 = needs_scatter or in2_features == 1
        grad_in1 = (
            (torch.zeros_like(in1) if _zero_g1 else torch.empty_like(in1))
            if need_grad_in1
            else torch.empty((0,), device=in1.device, dtype=in1.dtype)
        )
        grad_in2 = (
            (torch.zeros_like(in2) if _zero_g2 else torch.empty_like(in2))
            if need_grad_in2
            else torch.empty((0,), device=in2.device, dtype=in2.dtype)
        )
        if need_grad_weights:
            grad_weights = torch.zeros_like(weights) if shared_weights else torch.empty_like(weights)
        else:
            grad_weights = torch.empty((0,), device=weights.device, dtype=weights.dtype)
        grid = lambda meta: (
            triton.cdiv(n_batch, meta["BLOCK_SIZE"]),
            triton.cdiv(out_features, meta["FEATURE_BLOCK"]),
        )

        common_kwargs = dict(
            n_batch=n_batch,
            IN1_FEATURES=in1_features,
            IN2_FEATURES=in2_features,
            OUT_FEATURES=out_features,
            GRAD_OUT_STRIDE=grad_out.stride(0),
            GRAD_OUT_COL_STRIDE=grad_out.stride(1),
            in1_stride=in1.stride(0),
            in2_stride=in2.stride(0),
            IN1_L_MAX=in1_l_max,
            IN2_L_MAX=in2_l_max,
            OUT_L_MAX=out_l_max,
            SYMMETRIC_PRODUCT=int(symmetric_product),
            SHARED_WEIGHTS=int(shared_weights),
            **_per_l_kwargs("IN1_OFF_L", in1_offs),
            **_per_l_kwargs("IN2_OFF_L", in2_offs),
            **_per_l_kwargs("OUT_L", out_sizes, suffix="_SIZE"),
            **_per_l_kwargs("N_PATHS_L", n_paths_per_l),
            N_TOTAL_PATHS=n_total_paths,
            REDUCE_PATHS=int(reduce_paths),
        )

        if needs_scatter and TUNING.scatter_kernel == "edge":
            groups_l = _csr_path_groups(out_l_max, n_paths_per_l, reduce_paths, budget=TUNING.bwd_group_budget)
            n_groups = max(groups_l[: out_l_max + 1]) + 1
            plan = _resolve_split_plan("bwd", out_l_max)
            launches = [(m1 & need_grad_in1, m2 & need_grad_in2, mw & need_grad_weights) for m1, m2, mw in plan]
            launches = [lg for lg in launches if any(lg)]

            def _run_all_edge() -> None:
                for ni1, ni2, nw in launches:
                    for path_group in range(n_groups):
                        launcher(tp_bwd_kernel)[grid](
                            grad_out,
                            in1,
                            in2,
                            weights,
                            grad_in1,
                            grad_in2,
                            grad_weights,
                            idx_i_s,
                            idx_j_s,
                            grad_in1_stride=grad_in1.stride(0),
                            grad_in2_stride=grad_in2.stride(0),
                            NEED_GRAD_IN1=int(ni1),
                            NEED_GRAD_IN2=int(ni2),
                            NEED_GRAD_WEIGHTS=int(nw),
                            NEEDS_SCATTER=1,
                            PATH_GROUP=path_group,
                            N_PATH_GROUPS=n_groups,
                            **_per_l_kwargs("GROUP_L", groups_l),
                            **common_kwargs,
                        )

            _run_with_retune_guard(
                tp_bwd_kernel,
                _run_all_edge,
                (grad_in1, grad_in2, grad_weights),
                multi=len(launches) > 1 or n_groups > 1,
            )
            return grad_in1, grad_in2, grad_weights

        if needs_scatter:
            rowptr_i = torch.searchsorted(
                idx_i_s,
                torch.arange(out_batch + 1, device=idx_i_s.device, dtype=idx_i_s.dtype),
            )
            groups_l = _csr_path_groups(out_l_max, n_paths_per_l, reduce_paths, budget=TUNING.bwd_group_budget)
            n_groups = max(groups_l[: out_l_max + 1]) + 1
            n_slots = _bwd_csr_chunk_slots(n_batch, out_batch)
            csr_grid = lambda meta: (out_batch * n_slots, triton.cdiv(out_features, meta["FEATURE_BLOCK"]))
            csr_kwargs = {k: v for k, v in common_kwargs.items() if k != "n_batch"}

            plan = _resolve_split_plan("bwd", out_l_max)
            launches = [(m1 & need_grad_in1, m2 & need_grad_in2, mw & need_grad_weights) for m1, m2, mw in plan]
            launches = [lg for lg in launches if any(lg)]

            def _launch(ni1: int, ni2: int, nw: int, path_group: int) -> None:
                launcher(tp_bwd_csr_kernel)[csr_grid](
                    grad_out,
                    in1,
                    in2,
                    weights,
                    grad_in1,
                    grad_in2,
                    grad_weights,
                    idx_j_s,
                    rowptr_i,
                    grad_in1_stride=grad_in1.stride(0),
                    grad_in2_stride=grad_in2.stride(0),
                    N_CHUNK_SLOTS=n_slots,
                    NEED_GRAD_IN1=int(ni1),
                    NEED_GRAD_IN2=int(ni2),
                    NEED_GRAD_WEIGHTS=int(nw),
                    PATH_GROUP=path_group,
                    N_PATH_GROUPS=n_groups,
                    **_per_l_kwargs("GROUP_L", groups_l),
                    **csr_kwargs,
                )

            def _run_all() -> None:
                for ni1, ni2, nw in launches:
                    for path_group in range(n_groups):
                        _launch(ni1, ni2, nw, path_group)

            _run_with_retune_guard(
                tp_bwd_csr_kernel,
                _run_all,
                (grad_in1, grad_in2, grad_weights),
                multi=len(launches) > 1 or n_groups > 1,
            )
            return grad_in1, grad_in2, grad_weights

        plan = _SPLIT_PLANS["bwd"].get(TUNING.bwd_split_override or "fused", _SPLIT_PLANS["bwd"]["fused"])
        launches = [(m1 & need_grad_in1, m2 & need_grad_in2, mw & need_grad_weights) for m1, m2, mw in plan]
        launches = [lg for lg in launches if any(lg)]

        def _run_all_block() -> None:
            for ni1, ni2, nw in launches:
                launcher(tp_bwd_kernel)[grid](
                    grad_out,
                    in1,
                    in2,
                    weights,
                    grad_in1,
                    grad_in2,
                    grad_weights,
                    idx_i_s,
                    idx_j_s,
                    grad_in1_stride=grad_in1.stride(0),
                    grad_in2_stride=grad_in2.stride(0),
                    NEED_GRAD_IN1=int(ni1),
                    NEED_GRAD_IN2=int(ni2),
                    NEED_GRAD_WEIGHTS=int(nw),
                    NEEDS_SCATTER=0,
                    PATH_GROUP=0,
                    N_PATH_GROUPS=1,
                    **_per_l_kwargs("GROUP_L", [0] * _NUM_L_SLOTS),
                    **common_kwargs,
                )

        _run_with_retune_guard(
            tp_bwd_kernel,
            _run_all_block,
            (grad_in1, grad_in2, grad_weights),
            multi=len(launches) > 1,
        )
        return grad_in1, grad_in2, grad_weights

    def tp_bwd_op_setup_context(ctx, inputs, output):
        (
            grad_out,
            in1,
            in2,
            weights,
            idx_i,
            idx_j,
            in1_l_max,
            in2_l_max,
            out_l_max,
            in1_features,
            in2_features,
            symmetric_product,
            shared_weights,
            in1_offs,
            in2_offs,
            out_sizes,
            n_paths_per_l,
            n_total_paths,
            reduce_paths,
            need_grad_in1,
            need_grad_in2,
            need_grad_weights,
        ) = inputs
        if hasattr(ctx, "set_materialize_grads"):
            ctx.set_materialize_grads(False)
        ctx.need_grad_in1_output = bool(need_grad_in1)
        ctx.need_grad_in2_output = bool(need_grad_in2)
        ctx.need_grad_weights_output = bool(need_grad_weights)
        ctx.save_for_backward(grad_out, in1, in2, weights, idx_i, idx_j)
        _save_tp_scalars_to_ctx(
            ctx,
            in1_l_max,
            in2_l_max,
            out_l_max,
            in1_features,
            in2_features,
            symmetric_product,
            shared_weights,
            in1_offs,
            in2_offs,
            out_sizes,
            n_paths_per_l,
            n_total_paths,
            reduce_paths,
        )

    def tp_dbwd(
        ctx,
        v_in1: Optional[torch.Tensor],
        v_in2: Optional[torch.Tensor],
        v_weights: Optional[torch.Tensor],
    ):
        grad_out, in1, in2, weights, idx_i, idx_j = ctx.saved_tensors

        in1 = in1.contiguous()
        in2 = in2.contiguous()
        weights = weights.contiguous()
        if v_in1 is None or v_in1.numel() == 0:
            v_in1 = torch.zeros_like(in1)
        if v_in2 is None or v_in2.numel() == 0:
            v_in2 = torch.zeros_like(in2)
        has_v_weights = v_weights is not None and v_weights.numel() != 0
        if not has_v_weights:
            v_weights = weights
        v_weights = v_weights.contiguous()

        need_d_grad_out = bool(ctx.needs_input_grad[0])
        need_d_in1 = bool(ctx.needs_input_grad[1])
        need_d_in2 = bool(ctx.needs_input_grad[2])
        need_d_weights = bool(ctx.needs_input_grad[3])
        if not (need_d_grad_out or need_d_in1 or need_d_in2 or need_d_weights):
            return (None, None, None, None) + (None,) * (len(ctx.needs_input_grad) - 4)

        out_features = max(ctx.in1_features, ctx.in2_features)
        out_batch = grad_out.shape[0]
        needs_scatter, idx_i_s, idx_j_s, n_batch, _ = _resolve_scatter(idx_i, idx_j, in1)
        allocs = _alloc_out_sizes(
            ctx.reduce_paths,
            out_features,
            ctx.n_paths_per_l,
            ctx.out_sizes,
        )
        v_in1 = v_in1.contiguous()
        v_in2 = v_in2.contiguous()

        d_grad_out_alloc = torch.zeros if (needs_scatter or ctx.reduce_paths) else torch.empty
        d_grad_out = (
            d_grad_out_alloc((out_batch, sum(allocs)), device=in1.device, dtype=in1.dtype)
            if need_d_grad_out
            else torch.empty((0,), device=in1.device, dtype=in1.dtype)
        )
        _zero_d1 = needs_scatter or ctx.in1_features == 1
        _zero_d2 = needs_scatter or ctx.in2_features == 1
        d_in1 = (
            (torch.zeros_like(in1) if _zero_d1 else torch.empty_like(in1))
            if need_d_in1
            else torch.empty((0,), device=in1.device, dtype=in1.dtype)
        )
        d_in2 = (
            (torch.zeros_like(in2) if _zero_d2 else torch.empty_like(in2))
            if need_d_in2
            else torch.empty((0,), device=in2.device, dtype=in2.dtype)
        )
        if need_d_weights:
            d_weights = torch.zeros_like(weights) if ctx.shared_weights else torch.empty_like(weights)
        else:
            d_weights = torch.empty((0,), device=weights.device, dtype=weights.dtype)

        grid = lambda meta: (
            triton.cdiv(n_batch, meta["BLOCK_SIZE"]),
            triton.cdiv(out_features, meta["FEATURE_BLOCK"]),
        )

        common_kwargs = dict(
            n_batch=n_batch,
            IN1_FEATURES=ctx.in1_features,
            IN2_FEATURES=ctx.in2_features,
            OUT_FEATURES=out_features,
            GRAD_OUT_STRIDE=grad_out.stride(0),
            GRAD_OUT_COL_STRIDE=grad_out.stride(1),
            in1_stride=in1.stride(0),
            in2_stride=in2.stride(0),
            NEEDS_SCATTER=int(needs_scatter),
            IN1_L_MAX=ctx.in1_l_max,
            IN2_L_MAX=ctx.in2_l_max,
            OUT_L_MAX=ctx.out_l_max,
            SYMMETRIC_PRODUCT=int(ctx.symmetric_product),
            SHARED_WEIGHTS=int(ctx.shared_weights),
            **_per_l_kwargs("IN1_OFF_L", ctx.in1_offs),
            **_per_l_kwargs("IN2_OFF_L", ctx.in2_offs),
            **_per_l_kwargs("OUT_L", allocs, suffix="_SIZE"),
            **_per_l_kwargs("N_PATHS_L", ctx.n_paths_per_l),
            N_TOTAL_PATHS=ctx.n_total_paths,
            REDUCE_PATHS=int(ctx.reduce_paths),
            HAS_V_WEIGHTS=int(has_v_weights),
            d_grad_out_stride=d_grad_out.stride(0),
            d_in1_stride=d_in1.stride(0),
            d_in2_stride=d_in2.stride(0),
        )

        if needs_scatter and TUNING.scatter_kernel == "edge":
            plan = _resolve_split_plan("dbwd_edge", ctx.out_l_max)
            launches = [
                (m0 & need_d_grad_out, m1 & need_d_in1, m2 & need_d_in2, mw & need_d_weights) for m0, m1, m2, mw in plan
            ]
            launches = [lg for lg in launches if any(lg)]

            def _run_all_edge() -> None:
                for ndgo, ndi1, ndi2, ndw in launches:
                    launcher(tp_dbwd_kernel)[grid](
                        grad_out,
                        in1,
                        in2,
                        weights,
                        v_in1,
                        v_in2,
                        v_weights,
                        d_grad_out,
                        d_weights,
                        d_in1,
                        d_in2,
                        idx_i_s,
                        idx_j_s,
                        NEED_D_GRAD_OUT=int(ndgo),
                        NEED_D_IN1=int(ndi1),
                        NEED_D_IN2=int(ndi2),
                        NEED_D_WEIGHTS=int(ndw),
                        **common_kwargs,
                    )

            _run_with_retune_guard(
                tp_dbwd_kernel,
                _run_all_edge,
                (d_grad_out, d_in1, d_in2, d_weights),
                multi=len(launches) > 1,
            )
            return (
                d_grad_out if need_d_grad_out else None,
                d_in1 if need_d_in1 else None,
                d_in2 if need_d_in2 else None,
                d_weights if need_d_weights else None,
            ) + (None,) * (len(ctx.needs_input_grad) - 4)

        if needs_scatter:
            rowptr_i = torch.searchsorted(
                idx_i_s,
                torch.arange(out_batch + 1, device=idx_i_s.device, dtype=idx_i_s.dtype),
            )
            csr_grid = lambda meta: (out_batch, triton.cdiv(out_features, meta["FEATURE_BLOCK"]))
            csr_kwargs = {k: v for k, v in common_kwargs.items() if k not in ("n_batch", "NEEDS_SCATTER")}

            plan = _resolve_split_plan("dbwd_csr", ctx.out_l_max)
            launches = [
                (m0 & need_d_grad_out, m1 & need_d_in1, m2 & need_d_in2, mw & need_d_weights) for m0, m1, m2, mw in plan
            ]
            launches = [lg for lg in launches if any(lg)]

            def _dbwd_launch_groups(mask):
                ndgo, ndi1, ndi2, ndw = mask
                # The budget of fused dbwd is register-pressure bound, so keep
                # it fixed.
                fused = ndgo and (ndi1 or ndi2 or ndw)
                if ndgo and not (ndi1 or ndi2 or ndw):
                    budget = TUNING.csr_group_budget
                elif not ndgo:
                    budget = TUNING.bwd_group_budget
                elif TUNING.dbwd_group_budget is not None:
                    budget = TUNING.dbwd_group_budget
                else:
                    budget = min(TUNING.csr_group_budget, TUNING.bwd_group_budget) // 2
                groups_l = _csr_path_groups(
                    ctx.out_l_max,
                    ctx.n_paths_per_l,
                    ctx.reduce_paths,
                    budget=budget,
                    peak_aware=not fused,
                )
                return groups_l, max(groups_l[: ctx.out_l_max + 1]) + 1

            launch_groups = [_dbwd_launch_groups(mask) for mask in launches]

            def _run_all_csr() -> None:
                for (ndgo, ndi1, ndi2, ndw), (groups_l, n_groups) in zip(launches, launch_groups):
                    for path_group in range(n_groups):
                        launcher(tp_dbwd_csr_kernel)[csr_grid](
                            grad_out,
                            in1,
                            in2,
                            weights,
                            v_in1,
                            v_in2,
                            v_weights,
                            d_grad_out,
                            d_weights,
                            d_in1,
                            d_in2,
                            idx_j_s,
                            rowptr_i,
                            NEED_D_GRAD_OUT=int(ndgo),
                            NEED_D_IN1=int(ndi1),
                            NEED_D_IN2=int(ndi2),
                            NEED_D_WEIGHTS=int(ndw),
                            PATH_GROUP=path_group,
                            N_PATH_GROUPS=n_groups,
                            **_per_l_kwargs("GROUP_L", groups_l),
                            **csr_kwargs,
                        )

            _run_with_retune_guard(
                tp_dbwd_csr_kernel,
                _run_all_csr,
                (d_grad_out, d_in1, d_in2, d_weights),
                multi=len(launches) > 1 or any(n > 1 for _, n in launch_groups),
            )
            return (
                d_grad_out if need_d_grad_out else None,
                d_in1 if need_d_in1 else None,
                d_in2 if need_d_in2 else None,
                d_weights if need_d_weights else None,
            ) + (None,) * (len(ctx.needs_input_grad) - 4)

        plan = _SPLIT_PLANS["dbwd"].get(TUNING.dbwd_split_override or "fused", _SPLIT_PLANS["dbwd"]["fused"])
        launches = [
            (m0 & need_d_grad_out, m1 & need_d_in1, m2 & need_d_in2, mw & need_d_weights) for m0, m1, m2, mw in plan
        ]
        launches = [lg for lg in launches if any(lg)]

        def _run_all_dense() -> None:
            for ndgo, ndi1, ndi2, ndw in launches:
                launcher(tp_dbwd_kernel)[grid](
                    grad_out,
                    in1,
                    in2,
                    weights,
                    v_in1,
                    v_in2,
                    v_weights,
                    d_grad_out,
                    d_weights,
                    d_in1,
                    d_in2,
                    idx_i_s,
                    idx_j_s,
                    NEED_D_GRAD_OUT=int(ndgo),
                    NEED_D_IN1=int(ndi1),
                    NEED_D_IN2=int(ndi2),
                    NEED_D_WEIGHTS=int(ndw),
                    **common_kwargs,
                )

        _run_with_retune_guard(
            tp_dbwd_kernel,
            _run_all_dense,
            (d_grad_out, d_in1, d_in2, d_weights),
            multi=len(launches) > 1,
        )

        return (
            d_grad_out if need_d_grad_out else None,
            d_in1 if need_d_in1 else None,
            d_in2 if need_d_in2 else None,
            d_weights if need_d_weights else None,
        ) + (None,) * (len(ctx.needs_input_grad) - 4)

    tp_fwd_op.register_autograd(tp_bwd, setup_context=tp_setup_context)
    tp_bwd_op.register_autograd(tp_dbwd, setup_context=tp_bwd_op_setup_context)

    def triton_tensor_product(
        in1,
        in2,
        weights,
        idx_i,
        idx_j,
        in1_l_max,
        in2_l_max,
        out_l_max,
        in1_features,
        in2_features,
        in1_slices,
        in2_slices,
        symmetric_product,
        shared_weights,
        n_paths,
        reduce_paths=False,
    ):
        """Evaluate tensor products with the available kernel backend.

        The Triton implementation requires CUDA inputs and ranks within the generated
        limit. When the Triton backend was unavailable at import, this entry point
        uses ``py_tensor_product`` instead.

        Args:
            in1 (torch.Tensor): First-input features of shape ``(n, in1_dim)``.
                In convolution mode, rows correspond to nodes.
            in2 (torch.Tensor): Second-input features of shape ``(n, in2_dim)``,
                or ``(n_edges, in2_dim)`` in convolution mode.
            weights (torch.Tensor): Path weights with a flattened path and feature
                axis. Shared weights are one-dimensional. Other weights have one
                row per input row or edge, as described by ``TensorProduct.forward``.
            idx_i (torch.Tensor or None): Receiver indices of shape ``(n_edges,)``.
                Convolution requires both index arrays. None selects independent rows.
            idx_j (torch.Tensor or None): Sender indices of shape ``(n_edges,)``.
            in1_l_max (int): Maximum tensor rank of the first input.
            in2_l_max (int): Maximum tensor rank of the second input.
            out_l_max (int): Maximum output tensor rank.
            in1_features (int): Feature channels of the first input.
            in2_features (int): Feature channels of the second input.
            in1_slices (list): Start and exclusive stop offsets for first-input ranks.
            in2_slices (list): Start and exclusive stop offsets for second-input ranks.
            symmetric_product (bool): Retain only couplings with ``l1 >= l2``.
                The caller must provide identical inputs for a symmetric product.
            shared_weights (bool): Use one weight vector for all rows or edges.
            n_paths (Sequence[int]): Path count at each output tensor rank.
            reduce_paths (bool): Sum paths at each output rank.

        Returns:
            torch.Tensor: Packed output features with one row per first-input row.
                Within each rank, axes are component, path, and feature before
                flattening. With path reduction, the path axis is summed away.
        """
        max_l = max(in1_l_max, in2_l_max, out_l_max)
        if max_l > KERNEL_L_MAX:
            raise NotImplementedError(
                f"triton_tensor_product supports l_max <= {KERNEL_L_MAX}; "
                f"got in1_l_max={in1_l_max}, in2_l_max={in2_l_max}, "
                f"out_l_max={out_l_max}."
            )

        out_features = max(in1_features, in2_features)
        n_total_paths = int(sum(n_paths))
        n_paths_per_l = _pad_path_counts(n_paths, KERNEL_L_MAX)
        out_sizes = _out_block_sizes(n_paths_per_l, out_features, reduce_paths)
        in1_offs = _slice_offsets(in1_slices, in1_l_max, KERNEL_L_MAX)
        in2_offs = _slice_offsets(in2_slices, in2_l_max, KERNEL_L_MAX)

        return torch.ops.flashcart.tp_fwd(
            in1,
            in2,
            weights,
            idx_i,
            idx_j,
            in1_l_max,
            in2_l_max,
            out_l_max,
            in1_features,
            in2_features,
            symmetric_product,
            shared_weights,
            in1_offs,
            in2_offs,
            out_sizes,
            n_paths_per_l,
            n_total_paths,
            reduce_paths,
        )

else:

    def triton_tensor_product(
        in1,
        in2,
        weights,
        idx_i,
        idx_j,
        in1_l_max,
        in2_l_max,
        out_l_max,
        in1_features,
        in2_features,
        in1_slices,
        in2_slices,
        symmetric_product,
        shared_weights,
        n_paths,
        reduce_paths,
    ):
        """Evaluate tensor products with the available kernel backend.

        The Triton implementation requires CUDA inputs and ranks within the generated
        limit. When the Triton backend was unavailable at import, this entry point
        uses ``py_tensor_product`` instead.

        Args:
            in1 (torch.Tensor): First-input features of shape ``(n, in1_dim)``.
                In convolution mode, rows correspond to nodes.
            in2 (torch.Tensor): Second-input features of shape ``(n, in2_dim)``,
                or ``(n_edges, in2_dim)`` in convolution mode.
            weights (torch.Tensor): Path weights with a flattened path and feature
                axis. Shared weights are one-dimensional. Other weights have one
                row per input row or edge, as described by ``TensorProduct.forward``.
            idx_i (torch.Tensor or None): Receiver indices of shape ``(n_edges,)``.
                Convolution requires both index arrays. None selects independent rows.
            idx_j (torch.Tensor or None): Sender indices of shape ``(n_edges,)``.
            in1_l_max (int): Maximum tensor rank of the first input.
            in2_l_max (int): Maximum tensor rank of the second input.
            out_l_max (int): Maximum output tensor rank.
            in1_features (int): Feature channels of the first input.
            in2_features (int): Feature channels of the second input.
            in1_slices (list): Start and exclusive stop offsets for first-input ranks.
            in2_slices (list): Start and exclusive stop offsets for second-input ranks.
            symmetric_product (bool): Retain only couplings with ``l1 >= l2``.
                The caller must provide identical inputs for a symmetric product.
            shared_weights (bool): Use one weight vector for all rows or edges.
            n_paths (Sequence[int]): Path count at each output tensor rank.
            reduce_paths (bool): Sum paths at each output rank.

        Returns:
            torch.Tensor: Packed output features with one row per first-input row.
                Within each rank, axes are component, path, and feature before
                flattening. With path reduction, the path axis is summed away.
        """
        if not TRITON_AVAILABLE:
            warnings.warn(
                "Triton is not installed. Falling back to PyTorch implementation. "
                "For better performance with CUDA, install triton with 'pip install triton'.",
                UserWarning,
                stacklevel=2,
            )
        elif not in1.is_cuda:
            warnings.warn(
                "Triton implementation requires CUDA tensors, but got CPU tensors. "
                "Falling back to PyTorch implementation.",
                UserWarning,
                stacklevel=2,
            )
        else:
            warnings.warn(
                "Triton tensor product is not available (CUDA not available). "
                "Falling back to PyTorch implementation.",
                UserWarning,
                stacklevel=2,
            )

        return py_tensor_product(
            in1,
            in2,
            weights,
            idx_i,
            idx_j,
            in1_l_max,
            in2_l_max,
            out_l_max,
            in1_features,
            in2_features,
            in1_slices,
            in2_slices,
            symmetric_product,
            shared_weights,
            n_paths,
            reduce_paths,
        )


def py_tensor_product(
    in1: torch.Tensor,
    in2: torch.Tensor,
    weights: torch.Tensor,
    idx_i: Optional[torch.Tensor],
    idx_j: Optional[torch.Tensor],
    in1_l_max: int,
    in2_l_max: int,
    out_l_max: int,
    in1_features: int,
    in2_features: int,
    in1_slices: list,
    in2_slices: list,
    symmetric_product: bool,
    shared_weights: bool,
    n_paths: list,
    reduce_paths: bool = False,
) -> torch.Tensor:
    """Evaluate tensor products with generated PyTorch operations.

    Requests within the cached rank limit use the imported generated module.
    Higher ranks use a callable generated and cached for the full operation
    configuration. That callable determines its own slices and path counts,
    replacing ``in1_slices``, ``in2_slices``, and ``n_paths``.

    Args:
        in1 (torch.Tensor): First-input features of shape ``(n, in1_dim)``.
            In convolution mode, rows correspond to nodes.
        in2 (torch.Tensor): Second-input features of shape ``(n, in2_dim)``,
            or ``(n_edges, in2_dim)`` in convolution mode.
        weights (torch.Tensor): Path weights with a flattened path and feature
            axis. Shared weights are one-dimensional. Other weights have one
            row per input row or edge, as described by ``TensorProduct.forward``.
        idx_i (torch.Tensor or None): Receiver indices of shape ``(n_edges,)``.
            Convolution requires both index arrays. None selects independent rows.
        idx_j (torch.Tensor or None): Sender indices of shape ``(n_edges,)``.
        in1_l_max (int): Maximum tensor rank of the first input.
        in2_l_max (int): Maximum tensor rank of the second input.
        out_l_max (int): Maximum output tensor rank.
        in1_features (int): Feature channels of the first input.
        in2_features (int): Feature channels of the second input.
        in1_slices (list): Start and exclusive stop offsets for first-input ranks.
        in2_slices (list): Start and exclusive stop offsets for second-input ranks.
        symmetric_product (bool): Retain only couplings with ``l1 >= l2``.
            The caller must provide identical inputs for a symmetric product.
        shared_weights (bool): Use one weight vector for all rows or edges.
        n_paths (Sequence[int]): Path count at each output tensor rank.
        reduce_paths (bool, optional): Sum paths at each output rank.
            Default: False.

    Returns:
        torch.Tensor: Packed output features with one row per first-input row.
            Within each rank, axes are component, path, and feature before
            flattening. With path reduction, the path axis is summed away.
    """
    if max(in1_l_max, in2_l_max, out_l_max) <= KERNEL_L_MAX:
        return _PY_TENSOR_PRODUCT_FORWARD_MODULE.py_tensor_product_forward(
            in1,
            in2,
            weights,
            idx_i,
            idx_j,
            in1_l_max,
            in2_l_max,
            out_l_max,
            in1_features,
            in2_features,
            in1_slices,
            in2_slices,
            symmetric_product,
            shared_weights,
            n_paths,
            reduce_paths,
        )
    del in1_slices, in2_slices, n_paths  # superseded by fallback codegen
    fn = compile_py_tensor_product_forward(
        in1_l_max=in1_l_max,
        in2_l_max=in2_l_max,
        out_l_max=out_l_max,
        in1_features=in1_features,
        in2_features=in2_features,
        symmetric_product=symmetric_product,
        shared_weights=shared_weights,
        reduce_paths=reduce_paths,
    )
    return fn(in1, in2, weights, idx_i, idx_j)
