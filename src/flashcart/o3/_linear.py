"""Evaluate equivariant linear maps with PyTorch or generated Triton kernels.

The generated module uses the maximum tensor rank read at import. Metadata
describes the input and output blocks and their per-rank weight matrices.
The Triton backend registers forward and derivative operations with PyTorch.
Calls with more active tensor ranks than the generated module supports use
the PyTorch implementation.
"""
import warnings
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from flashcart.o3._codegen_cache import load_generated
from flashcart.o3._codegen_linear import build_triton_linear_module_source
from flashcart.o3.kernel_config import kernel_l_max
from flashcart.o3.utils import (
    get_irreps_shapes,
    get_irreps_slices,
    get_linear_triton_kernel_options,
)

try:
    import triton
    from torch.library import triton_op

    from flashcart.o3._triton_launch import launcher

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

KERNEL_L_MAX: int = kernel_l_max()
_NUM_L_SLOTS = KERNEL_L_MAX + 1

_GENERATED_MODULE_NAME = "flashcart.o3._linear_kernels"
_CODEGEN_FILE = Path(__file__).with_name("_codegen_linear.py")


def _out_dim(out_l_max: int, out_features: int) -> int:
    return sum((2 * l + 1) * out_features for l in range(out_l_max + 1))


def pack_linear_weights(weights: Sequence[torch.Tensor]) -> torch.Tensor:
    """Concatenate per-rank weight matrices into one contiguous vector.

    Args:
        weights (Sequence[torch.Tensor]): Nonempty sequence of weight matrices
            in increasing tensor rank.

    Returns:
        torch.Tensor: Flattened weights in rank order.

    Raises:
        ValueError: ``weights`` is empty.
    """
    if not weights:
        raise ValueError("weights must contain at least one tensor.")
    return torch.cat([weight.reshape(-1) for weight in weights], dim=0).contiguous()


def py_linear(
    x: torch.Tensor,
    weights: Sequence[torch.Tensor],
    in_l_max: int,
    out_l_max: int,
    in_features: int,
    out_features: int,
    in_paths: Sequence[int],
    scale: torch.Tensor,
) -> torch.Tensor:
    """Evaluate scaled per-rank linear maps with PyTorch.

    Args:
        x (torch.Tensor): Packed input features of shape ``(n, in_dim)``.
        weights (Sequence[torch.Tensor]): Weight matrices in increasing tensor
            rank, each of shape ``(out_features, in_features * in_paths[l])``.
        in_l_max (int): Maximum input tensor rank.
        out_l_max (int): Maximum output tensor rank.
        in_features (int): Input channels per path and tensor rank.
        out_features (int): Output channels per tensor rank.
        in_paths (Sequence[int]): Path count for each input tensor rank.
        scale (torch.Tensor): One weight scale per shared input and output rank.

    Returns:
        torch.Tensor: Output features of shape
            ``(n, (out_l_max + 1)**2 * out_features)``. Output ranks above the
            input rank limit are filled with zeros.
    """
    common_l_max = min(in_l_max, out_l_max)
    path_list = list(in_paths)

    in_slices = get_irreps_slices(in_l_max, in_features, path_list)
    in_shapes = get_irreps_shapes(in_l_max, in_features, path_list)
    out_block_dims = [(2 * l + 1) * out_features for l in range(out_l_max + 1)]

    n_batch = x.shape[0]
    outputs = []
    for l in range(common_l_max + 1):
        start, stop = in_slices[l]
        x_l = x[:, start:stop].view(n_batch, *in_shapes[l])
        out_l = F.linear(x_l, weights[l] * scale[l], None)
        outputs.append(out_l.reshape(n_batch, -1))

    zero_pad_dim = sum(out_block_dims[l] for l in range(common_l_max + 1, out_l_max + 1))
    if zero_pad_dim > 0:
        outputs.append(x.new_zeros(n_batch, zero_pad_dim))
    return torch.cat(outputs, dim=-1)


def make_linear_metadata(
    device: torch.device,
    in_l_max: int,
    out_l_max: int,
    in_features: int,
    out_features: int,
    in_paths: Sequence[int],
    weight_scale: Optional[float],
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Construct tensor-rank metadata and scales for the linear kernels.

    Metadata contains one row for each rank shared by the input and output. Its
    columns are the input offset, output offset, packed-weight offset, number of
    tensor components, input channels including paths, and output channels.

    Args:
        device (torch.device): Device for the metadata and scale tensors.
        in_l_max (int): Maximum input tensor rank.
        out_l_max (int): Maximum output tensor rank.
        in_features (int): Input channels per path and tensor rank.
        out_features (int): Output channels per tensor rank.
        in_paths (Sequence[int]): Path count for each input tensor rank.
        weight_scale (float or None): Common weight scale. None uses
            ``1 / sqrt(in_features * in_paths[l])`` at each shared rank.

    Returns:
        tuple[torch.Tensor, torch.Tensor, int, int]: Metadata of shape
            ``(min(in_l_max, out_l_max) + 1, 6)`` with dtype ``torch.int64``,
            one floating-point scale per shared rank, the total number of
            weights, and the flattened output width. Both tensors are on
            ``device``. Scales use the default floating-point dtype.
    """
    common_l_max = min(in_l_max, out_l_max)
    path_list = list(in_paths)
    in_slices = get_irreps_slices(in_l_max, in_features, path_list)
    out_slices = get_irreps_slices(out_l_max, out_features, [1] * (out_l_max + 1))
    if weight_scale is not None:
        scale_list = [float(weight_scale)] * (common_l_max + 1)
    else:
        scale_list = [(float(in_features * path_list[l]) ** -0.5) for l in range(common_l_max + 1)]

    meta = []
    weight_offset = 0
    for l in range(common_l_max + 1):
        m_l = 2 * l + 1
        f_in_l = in_features * path_list[l]
        f_out_l = out_features
        meta.append([in_slices[l][0], out_slices[l][0], weight_offset, m_l, f_in_l, f_out_l])
        weight_offset += f_in_l * f_out_l

    meta_t = torch.tensor(meta, dtype=torch.int64, device=device)
    scale = torch.tensor(scale_list, device=device)
    out_size = _out_dim(out_l_max, out_features)
    return meta_t, scale, weight_offset, out_size


def _rowwise_grid(n_batch: int, max_m: int, max_o: int, n_active_l: int):
    def grid(META):
        return (
            triton.cdiv(n_batch * max_m, META["BLOCK_ROWS"]),
            triton.cdiv(max_o, META["BLOCK_OUT"]),
            n_active_l,
        )

    return grid


def _wgrad_grid(f_out_values: Sequence[int], f_in_values: Sequence[int], split_n: int, n_active_l: int):
    fo_fi = [(int(fo), int(fi)) for fo, fi in zip(f_out_values, f_in_values)]

    def grid(META):
        tiles = max(triton.cdiv(fo, META["BLOCK_FOUT"]) * triton.cdiv(fi, META["BLOCK_FIN"]) for fo, fi in fo_fi)
        return (tiles, split_n, n_active_l)

    return grid


def _slot_kwargs(prefix: str, values: Sequence[int], pad: int = 0, suffix: str = "") -> dict:
    vals = [int(v) for v in values]
    if len(vals) > _NUM_L_SLOTS:
        raise RuntimeError(f"Linear kernels support at most {_NUM_L_SLOTS} l-blocks, got {len(vals)}.")
    vals += [pad] * (_NUM_L_SLOTS - len(vals))
    return {f"{prefix}{i}{suffix}": vals[i] for i in range(_NUM_L_SLOTS)}


_SLOT_KWARGS_CACHE: dict = {}


def _rowwise_slot_kwargs(
    a_offsets: Sequence[int],
    o_offsets: Sequence[int],
    k_values: Sequence[int],
    o_values: Sequence[int],
    m_values: Sequence[int],
) -> dict:
    key = ("rowwise", tuple(a_offsets), tuple(o_offsets), tuple(k_values), tuple(o_values), tuple(m_values))
    cached = _SLOT_KWARGS_CACHE.get(key)
    if cached is None:
        cached = {
            **_slot_kwargs("A_OFF", a_offsets),
            **_slot_kwargs("O_OFF", o_offsets),
            **_slot_kwargs("K", k_values),
            **_slot_kwargs("O", o_values, pad=1),
            **_slot_kwargs("M", m_values, pad=1),
        }
        _SLOT_KWARGS_CACHE[key] = cached
    return cached


def _wgrad_slot_kwargs(
    in_offsets: Sequence[int],
    out_offsets: Sequence[int],
    f_in_values: Sequence[int],
    f_out_values: Sequence[int],
    m_values: Sequence[int],
) -> dict:
    key = ("wgrad", tuple(in_offsets), tuple(out_offsets), tuple(f_in_values), tuple(f_out_values), tuple(m_values))
    cached = _SLOT_KWARGS_CACHE.get(key)
    if cached is None:
        cached = {
            **_slot_kwargs("IN_OFF", in_offsets),
            **_slot_kwargs("OUT_OFF", out_offsets),
            **_slot_kwargs("FIN", f_in_values),
            **_slot_kwargs("FOUT", f_out_values, pad=1),
            **_slot_kwargs("M", m_values, pad=1),
        }
        _SLOT_KWARGS_CACHE[key] = cached
    return cached


def _slot_tensor_kwargs(prefix: str, tensors: Sequence[torch.Tensor], dummy: torch.Tensor) -> dict:
    ts = list(tensors)
    if len(ts) > _NUM_L_SLOTS:
        raise RuntimeError(f"Linear kernels support at most {_NUM_L_SLOTS} l-blocks, got {len(ts)}.")
    ts += [dummy] * (_NUM_L_SLOTS - len(ts))
    return {f"{prefix}{i}_ptr": ts[i] for i in range(_NUM_L_SLOTS)}


_DEVICE_SM_COUNT: dict = {}


def _sm_count(device: torch.device) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    count = _DEVICE_SM_COUNT.get(idx)
    if count is None:
        count = torch.cuda.get_device_properties(idx).multi_processor_count
        _DEVICE_SM_COUNT[idx] = count
    return count


def _wgrad_split_n(
    n_batch: int,
    m_values: Sequence[int],
    f_in_values: Sequence[int],
    f_out_values: Sequence[int],
    device: torch.device,
) -> int:
    tiles = sum(-(-int(fo) // 64) * -(-int(fi) // 64) for fo, fi in zip(f_out_values, f_in_values))
    if n_batch == 0 or tiles == 0:
        return 1
    target = 4 * _sm_count(device)
    if tiles >= target:
        return 1
    rows = n_batch * max(int(v) for v in m_values)
    want = -(-target // tiles)
    split = min(1 << max(0, want - 1).bit_length() if want > 1 else 1, max(1, rows // 512), 32)
    return max(1, split)


MetaLists = Tuple[Sequence[int], Sequence[int], Sequence[int], Sequence[int], Sequence[int], Sequence[int]]


def _n_active_l(meta: torch.Tensor, meta_lists: Optional[MetaLists]) -> int:
    if meta_lists is not None:
        return len(meta_lists[0])
    return int(meta.shape[0])


def _prepare_linear_weights(
    weights: Sequence[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    f_in_values: Sequence[int],
    f_out_values: Sequence[int],
    n_active_l: int,
) -> List[torch.Tensor]:
    if len(weights) < n_active_l:
        raise RuntimeError(f"Expected at least {n_active_l} weight tensors, got {len(weights)}.")
    out: List[torch.Tensor] = []
    for l in range(n_active_l):
        f_in_l = int(f_in_values[l])
        f_out_l = int(f_out_values[l])
        w = weights[l]
        if w.shape != (f_out_l, f_in_l):
            raise RuntimeError(f"Weight[{l}] shape mismatch: expected ({f_out_l}, {f_in_l}), got {tuple(w.shape)}.")
        if w.device != device or w.dtype != dtype:
            w = w.to(device=device, dtype=dtype)
        out.append(w)
    return out


if TRITON_AVAILABLE and torch.cuda.is_available():

    _kernels = load_generated(_GENERATED_MODULE_NAME, _CODEGEN_FILE, KERNEL_L_MAX, build_triton_linear_module_source)

    _linear_rowwise_kernel = _kernels.linear_rowwise_kernel
    _linear_wgrad_kernel = _kernels.linear_wgrad_kernel

    def _contiguous_list(tensors: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        return [t if t.is_contiguous() else t.contiguous() for t in tensors]

    def _active_end(offsets: Sequence[int], m_values: Sequence[int], f_values: Sequence[int], n_active_l: int) -> int:
        l_last = n_active_l - 1
        return int(offsets[l_last]) + int(m_values[l_last]) * int(f_values[l_last])

    @triton_op("flashcart::linear_fwd", mutates_args={})
    def linear_fwd_op(
        x: torch.Tensor,
        weights: List[torch.Tensor],
        meta: torch.Tensor,
        out_size: int,
        use_fp64: bool,
        dot_input_precision: str,
        in_offsets: List[int],
        out_offsets: List[int],
        weight_offsets: List[int],
        m_values: List[int],
        f_in_values: List[int],
        f_out_values: List[int],
        scale: torch.Tensor,
    ) -> torch.Tensor:
        x = x.contiguous()
        n_batch = x.shape[0]
        meta_lists = (
            in_offsets,
            out_offsets,
            weight_offsets,
            m_values,
            f_in_values,
            f_out_values,
        )
        n_active_l = _n_active_l(meta, meta_lists)
        if n_active_l == 0:
            return x.new_zeros((n_batch, int(out_size)))

        out = x.new_empty((n_batch, int(out_size)))
        active_out_end = _active_end(out_offsets, m_values, f_out_values, n_active_l)
        if active_out_end < int(out_size):
            out[:, active_out_end:].zero_()

        dummy = x.new_empty((1,))
        max_m = max(int(v) for v in m_values)
        max_o = max(int(v) for v in f_out_values)
        launcher(_linear_rowwise_kernel)[_rowwise_grid(n_batch, max_m, max_o, n_active_l)](
            a_ptr=x,
            b_ptr=dummy,
            out_ptr=out,
            scale_ptr=scale,
            **_slot_tensor_kwargs("w", _contiguous_list(weights[:n_active_l]), dummy),
            **_slot_tensor_kwargs("v", [], dummy),
            n_batch=n_batch,
            a_stride_n=x.stride(0),
            b_stride_n=0,
            out_stride_n=out.stride(0),
            N_ACTIVE_L=n_active_l,
            **_rowwise_slot_kwargs(in_offsets, out_offsets, f_in_values, f_out_values, m_values),
            TERM1=True,
            TERM2=False,
            TRANS_W=True,
            USE_FP64=use_fp64,
            INPUT_PRECISION=dot_input_precision,
        )
        return out

    def linear_setup_context(ctx, inputs, output):
        (
            x,
            weights,
            meta,
            out_size,
            use_fp64,
            dot_input_precision,
            in_offsets,
            out_offsets,
            weight_offsets,
            m_values,
            f_in_values,
            f_out_values,
            scale,
        ) = inputs
        del out_size, meta
        ctx.save_for_backward(x, *weights, scale)
        ctx.use_fp64 = bool(use_fp64)
        ctx.dot_input_precision = dot_input_precision
        ctx.in_offsets = list(in_offsets)
        ctx.out_offsets = list(out_offsets)
        ctx.weight_offsets = list(weight_offsets)
        ctx.m_values = list(m_values)
        ctx.f_in_values = list(f_in_values)
        ctx.f_out_values = list(f_out_values)

    def linear_bwd(ctx, grad_out: torch.Tensor):
        saved = ctx.saved_tensors
        x = saved[0]
        scale = saved[-1]
        weights = list(saved[1:-1])
        need_x = bool(ctx.needs_input_grad[0])
        need_weight = bool(ctx.needs_input_grad[1])
        if need_x or need_weight:
            grad_x, grad_weights = torch.ops.flashcart.linear_bwd(
                grad_out,
                x,
                weights,
                need_x,
                need_weight,
                ctx.use_fp64,
                ctx.dot_input_precision,
                ctx.in_offsets,
                ctx.out_offsets,
                ctx.weight_offsets,
                ctx.m_values,
                ctx.f_in_values,
                ctx.f_out_values,
                scale,
            )
        else:
            grad_x = None
            grad_weights = None
        return (
            grad_x,
            grad_weights,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    @triton_op("flashcart::linear_bwd", mutates_args={})
    def linear_bwd_op(
        grad_out: torch.Tensor,
        x: torch.Tensor,
        weights: List[torch.Tensor],
        need_x: bool,
        need_weight: bool,
        use_fp64: bool,
        dot_input_precision: str,
        in_offsets: List[int],
        out_offsets: List[int],
        weight_offsets: List[int],
        m_values: List[int],
        f_in_values: List[int],
        f_out_values: List[int],
        scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        grad_out = grad_out.contiguous()
        x = x.contiguous()
        n_batch = x.shape[0]
        n_active_l = len(in_offsets)

        if n_active_l == 0 or not (need_x or need_weight):
            grad_x = torch.zeros_like(x) if need_x else x.new_empty((0,))
            grad_weights = [torch.zeros_like(w) for w in weights] if need_weight else [w.new_empty(0) for w in weights]
            return grad_x, grad_weights

        dummy = x.new_empty((1,))
        max_m = max(int(v) for v in m_values)

        if need_x:
            grad_x = torch.empty_like(x)
            active_in_end = _active_end(in_offsets, m_values, f_in_values, n_active_l)
            if active_in_end < x.shape[1]:
                grad_x[:, active_in_end:].zero_()
            max_fin = max(int(v) for v in f_in_values)
            launcher(_linear_rowwise_kernel)[_rowwise_grid(n_batch, max_m, max_fin, n_active_l)](
                a_ptr=grad_out,
                b_ptr=dummy,
                out_ptr=grad_x,
                scale_ptr=scale,
                **_slot_tensor_kwargs("w", _contiguous_list(weights), dummy),
                **_slot_tensor_kwargs("v", [], dummy),
                n_batch=n_batch,
                a_stride_n=grad_out.stride(0),
                b_stride_n=0,
                out_stride_n=grad_x.stride(0),
                N_ACTIVE_L=n_active_l,
                **_rowwise_slot_kwargs(out_offsets, in_offsets, f_out_values, f_in_values, m_values),
                TERM1=True,
                TERM2=False,
                TRANS_W=False,
                USE_FP64=use_fp64,
                INPUT_PRECISION=dot_input_precision,
            )
        else:
            grad_x = x.new_empty((0,))

        if need_weight:
            split_n = _wgrad_split_n(n_batch, m_values, f_in_values, f_out_values, x.device)
            if split_n > 1:
                grad_weights = [torch.zeros_like(w, memory_format=torch.contiguous_format) for w in weights]
            else:
                grad_weights = [torch.empty_like(w, memory_format=torch.contiguous_format) for w in weights]
            launcher(_linear_wgrad_kernel)[_wgrad_grid(f_out_values, f_in_values, split_n, n_active_l)](
                go_ptr=grad_out,
                b_ptr=x,
                scale_ptr=scale,
                **_slot_tensor_kwargs("g", grad_weights, dummy),
                n_batch=n_batch,
                go_stride_n=grad_out.stride(0),
                b_stride_n=x.stride(0),
                N_ACTIVE_L=n_active_l,
                **_wgrad_slot_kwargs(in_offsets, out_offsets, f_in_values, f_out_values, m_values),
                SPLIT=split_n > 1,
                USE_FP64=use_fp64,
                INPUT_PRECISION=dot_input_precision,
            )
        else:
            grad_weights = [w.new_empty(0) for w in weights]

        return grad_x, grad_weights

    def linear_bwd_setup_context(ctx, inputs, output):
        (
            grad_out,
            x,
            weights,
            need_x,
            need_weight,
            use_fp64,
            dot_input_precision,
            in_offsets,
            out_offsets,
            weight_offsets,
            m_values,
            f_in_values,
            f_out_values,
            scale,
        ) = inputs
        if hasattr(ctx, "set_materialize_grads"):
            ctx.set_materialize_grads(False)
        ctx.save_for_backward(grad_out, x, *weights, scale)
        ctx.need_x_output = bool(need_x)
        ctx.need_weight_output = bool(need_weight)
        ctx.use_fp64 = bool(use_fp64)
        ctx.dot_input_precision = dot_input_precision
        ctx.in_offsets = list(in_offsets)
        ctx.out_offsets = list(out_offsets)
        ctx.weight_offsets = list(weight_offsets)
        ctx.m_values = list(m_values)
        ctx.f_in_values = list(f_in_values)
        ctx.f_out_values = list(f_out_values)

    @triton_op("flashcart::linear_dbwd", mutates_args={})
    def linear_dbwd_op(
        grad_out: torch.Tensor,
        x: torch.Tensor,
        weights: List[torch.Tensor],
        v_grad_x: torch.Tensor,
        v_grad_weight: List[torch.Tensor],
        use_fp64: bool,
        dot_input_precision: str,
        need_d_grad_out: bool,
        need_d_x: bool,
        need_d_weight: bool,
        has_v_grad_x: bool,
        has_v_grad_weight: bool,
        in_offsets: List[int],
        out_offsets: List[int],
        weight_offsets: List[int],
        m_values: List[int],
        f_in_values: List[int],
        f_out_values: List[int],
        scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        grad_out = grad_out.contiguous()
        x = x.contiguous()
        n_batch = x.shape[0]
        out_dim = grad_out.shape[1]
        n_active_l = len(in_offsets)

        run_dgrad = need_d_grad_out and (has_v_grad_x or has_v_grad_weight) and n_active_l > 0
        run_d_x = need_d_x and has_v_grad_weight and n_active_l > 0
        run_d_weight = need_d_weight and has_v_grad_x and n_active_l > 0

        dummy = x.new_empty((1,))
        max_m = max((int(v) for v in m_values), default=1)

        if has_v_grad_x:
            v_grad_x = v_grad_x.contiguous()
        if has_v_grad_weight:
            v_grad_weight = _contiguous_list(v_grad_weight)

        if need_d_grad_out:
            if run_dgrad:
                d_grad_out = grad_out.new_empty((n_batch, out_dim))
                active_out_end = _active_end(out_offsets, m_values, f_out_values, n_active_l)
                if active_out_end < out_dim:
                    d_grad_out[:, active_out_end:].zero_()
                max_o = max(int(v) for v in f_out_values)
                launcher(_linear_rowwise_kernel)[_rowwise_grid(n_batch, max_m, max_o, n_active_l)](
                    a_ptr=v_grad_x if has_v_grad_x else dummy,
                    b_ptr=x,
                    out_ptr=d_grad_out,
                    scale_ptr=scale,
                    **_slot_tensor_kwargs("w", _contiguous_list(weights) if has_v_grad_x else [], dummy),
                    **_slot_tensor_kwargs("v", v_grad_weight if has_v_grad_weight else [], dummy),
                    n_batch=n_batch,
                    a_stride_n=v_grad_x.stride(0) if has_v_grad_x else 0,
                    b_stride_n=x.stride(0),
                    out_stride_n=d_grad_out.stride(0),
                    N_ACTIVE_L=n_active_l,
                    **_rowwise_slot_kwargs(in_offsets, out_offsets, f_in_values, f_out_values, m_values),
                    TERM1=has_v_grad_x,
                    TERM2=has_v_grad_weight,
                    TRANS_W=True,
                    USE_FP64=use_fp64,
                    INPUT_PRECISION=dot_input_precision,
                )
            else:
                d_grad_out = grad_out.new_zeros((n_batch, out_dim))
        else:
            d_grad_out = grad_out.new_empty((0,))

        if need_d_x:
            if run_d_x:
                d_x = torch.empty_like(x)
                active_in_end = _active_end(in_offsets, m_values, f_in_values, n_active_l)
                if active_in_end < x.shape[1]:
                    d_x[:, active_in_end:].zero_()
                max_fin = max(int(v) for v in f_in_values)
                launcher(_linear_rowwise_kernel)[_rowwise_grid(n_batch, max_m, max_fin, n_active_l)](
                    a_ptr=grad_out,
                    b_ptr=dummy,
                    out_ptr=d_x,
                    scale_ptr=scale,
                    **_slot_tensor_kwargs("w", v_grad_weight, dummy),
                    **_slot_tensor_kwargs("v", [], dummy),
                    n_batch=n_batch,
                    a_stride_n=grad_out.stride(0),
                    b_stride_n=0,
                    out_stride_n=d_x.stride(0),
                    N_ACTIVE_L=n_active_l,
                    **_rowwise_slot_kwargs(out_offsets, in_offsets, f_out_values, f_in_values, m_values),
                    TERM1=True,
                    TERM2=False,
                    TRANS_W=False,
                    USE_FP64=use_fp64,
                    INPUT_PRECISION=dot_input_precision,
                )
            else:
                d_x = torch.zeros_like(x)
        else:
            d_x = x.new_empty((0,))

        if need_d_weight:
            if run_d_weight:
                split_n = _wgrad_split_n(n_batch, m_values, f_in_values, f_out_values, x.device)
                if split_n > 1:
                    d_weights = [torch.zeros_like(w, memory_format=torch.contiguous_format) for w in weights]
                else:
                    d_weights = [torch.empty_like(w, memory_format=torch.contiguous_format) for w in weights]
                launcher(_linear_wgrad_kernel)[_wgrad_grid(f_out_values, f_in_values, split_n, n_active_l)](
                    go_ptr=grad_out,
                    b_ptr=v_grad_x,
                    scale_ptr=scale,
                    **_slot_tensor_kwargs("g", d_weights, dummy),
                    n_batch=n_batch,
                    go_stride_n=grad_out.stride(0),
                    b_stride_n=v_grad_x.stride(0),
                    N_ACTIVE_L=n_active_l,
                    **_wgrad_slot_kwargs(in_offsets, out_offsets, f_in_values, f_out_values, m_values),
                    SPLIT=split_n > 1,
                    USE_FP64=use_fp64,
                    INPUT_PRECISION=dot_input_precision,
                )
            else:
                d_weights = [torch.zeros_like(w) for w in weights]
        else:
            d_weights = [w.new_empty(0) for w in weights]

        return d_grad_out, d_x, d_weights

    def linear_dbwd(
        ctx,
        v_grad_x: Optional[torch.Tensor],
        v_grad_weight: Optional[List[torch.Tensor]],
    ):
        saved = ctx.saved_tensors
        grad_out = saved[0]
        x = saved[1]
        scale = saved[-1]
        weights = list(saved[2:-1])
        need_d_grad_out = bool(ctx.needs_input_grad[0])
        need_d_x = bool(ctx.needs_input_grad[1])
        need_d_weight = bool(ctx.needs_input_grad[2])
        if not (need_d_grad_out or need_d_x or need_d_weight):
            return (None,) * 14

        has_v_grad_x = v_grad_x is not None and v_grad_x.numel() != 0 and ctx.need_x_output
        has_v_grad_weight = (
            v_grad_weight is not None
            and len(v_grad_weight) > 0
            and any(v is not None and v.numel() != 0 for v in v_grad_weight)
            and ctx.need_weight_output
        )

        v_grad_x_arg = v_grad_x if has_v_grad_x else x.new_empty((0,))
        if has_v_grad_weight:
            v_grad_weight_arg = [
                (
                    (v.contiguous() if not v.is_contiguous() else v)
                    if v is not None and v.numel() != 0
                    else torch.zeros_like(w, memory_format=torch.contiguous_format)
                )
                for v, w in zip(v_grad_weight, weights, strict=True)
            ]
        else:
            v_grad_weight_arg = [w.new_empty(0) for w in weights]

        d_grad_out, d_x, d_weights = torch.ops.flashcart.linear_dbwd(
            grad_out,
            x,
            weights,
            v_grad_x_arg,
            v_grad_weight_arg,
            ctx.use_fp64,
            ctx.dot_input_precision,
            need_d_grad_out,
            need_d_x,
            need_d_weight,
            has_v_grad_x,
            has_v_grad_weight,
            ctx.in_offsets,
            ctx.out_offsets,
            ctx.weight_offsets,
            ctx.m_values,
            ctx.f_in_values,
            ctx.f_out_values,
            scale,
        )

        return (
            d_grad_out if need_d_grad_out else None,
            d_x if need_d_x else None,
            d_weights if need_d_weight else None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    linear_fwd_op.register_autograd(linear_bwd, setup_context=linear_setup_context)
    linear_bwd_op.register_autograd(linear_dbwd, setup_context=linear_bwd_setup_context)

    def triton_linear(
        x: torch.Tensor,
        weights: Sequence[torch.Tensor],
        in_l_max: int,
        out_l_max: int,
        in_features: int,
        out_features: int,
        *,
        in_paths: Sequence[int],
        scale: torch.Tensor,
        precomputed_meta: Optional[torch.Tensor] = None,
        precomputed_meta_lists: Optional[MetaLists] = None,
    ) -> torch.Tensor:
        """Evaluate scaled per-rank linear maps with the available kernel backend.

        CUDA inputs use Triton when available and when the active ranks fit the
        configured kernel limit. Other calls use ``py_linear``. Precomputed metadata
        and its Python lists must describe the same layout.

        Args:
            x (torch.Tensor): Packed input features of shape ``(n, in_dim)``.
            weights (Sequence[torch.Tensor]): Weight matrices in increasing tensor
                rank, each of shape ``(out_features, in_features * in_paths[l])``.
            in_l_max (int): Maximum input tensor rank.
            out_l_max (int): Maximum output tensor rank.
            in_features (int): Input channels per path and tensor rank.
            out_features (int): Output channels per tensor rank.
            in_paths (Sequence[int]): Path count for each input tensor rank.
            scale (torch.Tensor): One weight scale per shared input and output rank.
            precomputed_meta (torch.Tensor, optional): Integer layout metadata from
                ``make_linear_metadata``. Default: None, which constructs metadata.
            precomputed_meta_lists (MetaLists, optional): Six Python lists containing
                the metadata columns. Required with ``precomputed_meta`` to avoid
                copying tensor metadata to CPU memory. Default: None.

        Returns:
            torch.Tensor: Output features of shape
                ``(n, (out_l_max + 1)**2 * out_features)``. Output ranks above the
                input rank limit are filled with zeros.
        """
        if not x.is_cuda:
            warnings.warn(
                "Triton linear requires CUDA tensors, but got CPU tensors. " "Falling back to PyTorch implementation.",
                UserWarning,
                stacklevel=2,
            )
            return py_linear(
                x,
                weights,
                in_l_max,
                out_l_max,
                in_features,
                out_features,
                in_paths=in_paths,
                scale=scale,
            )

        if x.dim() != 2:
            raise ValueError(f"x must be 2D [N, packed_dim], got shape {tuple(x.shape)}.")
        if not x.is_contiguous():
            x = x.contiguous()

        device = x.device
        dtype = x.dtype
        use_fp64, dot_input_precision = get_linear_triton_kernel_options(dtype)
        if precomputed_meta is not None:
            meta = precomputed_meta.to(device=device)
            if precomputed_meta_lists is None:
                raise ValueError("precomputed_meta_lists must be provided when precomputed_meta is used.")
            in_offsets, out_offsets, weight_offsets, m_values, f_in_values, f_out_values = (
                [int(v) for v in vals] for vals in precomputed_meta_lists
            )
            out_size = _out_dim(out_l_max, out_features)
        else:
            meta, _, _n_weight, out_size = make_linear_metadata(
                device=device,
                in_l_max=in_l_max,
                out_l_max=out_l_max,
                in_features=in_features,
                out_features=out_features,
                in_paths=in_paths,
                weight_scale=None,
            )
            rows = meta.tolist()
            in_offsets = [int(row[0]) for row in rows]
            out_offsets = [int(row[1]) for row in rows]
            weight_offsets = [int(row[2]) for row in rows]
            m_values = [int(row[3]) for row in rows]
            f_in_values = [int(row[4]) for row in rows]
            f_out_values = [int(row[5]) for row in rows]
        n_active_l = len(in_offsets)
        if n_active_l > _NUM_L_SLOTS:
            warnings.warn(
                f"Triton linear supports at most {_NUM_L_SLOTS} l-blocks, got {n_active_l}. "
                "Falling back to PyTorch implementation.",
                UserWarning,
                stacklevel=2,
            )
            return py_linear(
                x,
                weights,
                in_l_max,
                out_l_max,
                in_features,
                out_features,
                in_paths=in_paths,
                scale=scale,
            )
        weight_list = _prepare_linear_weights(
            weights,
            device=device,
            dtype=dtype,
            f_in_values=f_in_values,
            f_out_values=f_out_values,
            n_active_l=n_active_l,
        )

        return torch.ops.flashcart.linear_fwd(
            x,
            weight_list,
            meta,
            int(out_size),
            use_fp64,
            dot_input_precision,
            in_offsets,
            out_offsets,
            weight_offsets,
            m_values,
            f_in_values,
            f_out_values,
            scale,
        )

else:

    def triton_linear(
        x: torch.Tensor,
        weights: Sequence[torch.Tensor],
        in_l_max: int,
        out_l_max: int,
        in_features: int,
        out_features: int,
        *,
        in_paths: Sequence[int],
        scale: torch.Tensor,
        precomputed_meta: Optional[torch.Tensor] = None,
        precomputed_meta_lists: Optional[MetaLists] = None,
    ) -> torch.Tensor:
        """Evaluate scaled per-rank linear maps with the available kernel backend.

        CUDA inputs use Triton when available and when the active ranks fit the
        configured kernel limit. Other calls use ``py_linear``. Precomputed metadata
        and its Python lists must describe the same layout.

        Args:
            x (torch.Tensor): Packed input features of shape ``(n, in_dim)``.
            weights (Sequence[torch.Tensor]): Weight matrices in increasing tensor
                rank, each of shape ``(out_features, in_features * in_paths[l])``.
            in_l_max (int): Maximum input tensor rank.
            out_l_max (int): Maximum output tensor rank.
            in_features (int): Input channels per path and tensor rank.
            out_features (int): Output channels per tensor rank.
            in_paths (Sequence[int]): Path count for each input tensor rank.
            scale (torch.Tensor): One weight scale per shared input and output rank.
            precomputed_meta (torch.Tensor, optional): Integer layout metadata from
                ``make_linear_metadata``. Default: None, which constructs metadata.
            precomputed_meta_lists (MetaLists, optional): Six Python lists containing
                the metadata columns. Required with ``precomputed_meta`` to avoid
                copying tensor metadata to CPU memory. Default: None.

        Returns:
            torch.Tensor: Output features of shape
                ``(n, (out_l_max + 1)**2 * out_features)``. Output ranks above the
                input rank limit are filled with zeros.
        """
        del precomputed_meta, precomputed_meta_lists
        warnings.warn(
            "Triton linear kernels require CUDA and an installed triton package. "
            "Falling back to PyTorch implementation.",
            UserWarning,
            stacklevel=2,
        )
        return py_linear(
            x,
            weights,
            in_l_max,
            out_l_max,
            in_features,
            out_features,
            in_paths=in_paths,
            scale=scale,
        )
