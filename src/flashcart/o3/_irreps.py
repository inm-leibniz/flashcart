"""Evaluate Cartesian tensor expansions with generated PyTorch and Triton code.

Generated modules use the maximum tensor rank read at import. The Triton
backend registers expansion and derivative operations with PyTorch. PyTorch
evaluation generates additional expressions when the requested rank exceeds
the cached module's limit.
"""

import warnings
from pathlib import Path
from typing import Tuple

import torch

from flashcart.o3._codegen_cache import load_generated
from flashcart.o3._codegen_irreps import (
    build_py_irreps_forward_module_source,
    build_triton_irreps_module_source,
    compile_py_irreps_forward,
)
from flashcart.o3.kernel_config import kernel_l_max

try:
    import triton
    import triton.language as tl  # noqa: F401  (re-exported for downstream imports)
    from torch.library import triton_op

    from flashcart.o3._triton_launch import launcher

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

KERNEL_L_MAX: int = kernel_l_max()

_GENERATED_MODULE_NAME = "flashcart.o3._irreps_kernels"
_PY_FORWARD_MODULE_NAME = "flashcart.o3._py_irreps_forward"
_CODEGEN_FILE = Path(__file__).with_name("_codegen_irreps.py")


def _output_dim(l_max: int) -> int:
    return (l_max + 1) ** 2  # sum_l (2l + 1)


_PY_IRREPS_FORWARD_MODULE = load_generated(
    _PY_FORWARD_MODULE_NAME, _CODEGEN_FILE, KERNEL_L_MAX, build_py_irreps_forward_module_source
)


def py_irreps(x: "torch.Tensor", l_max: int) -> "torch.Tensor":
    """Evaluate the Cartesian expansion with generated PyTorch operations.

    Input vectors are expected to have unit length. This helper does not normalize
    or validate them. Ranks above the cached module's limit use a separately
    generated callable cached for the requested maximum rank.

    Args:
        x (torch.Tensor): Unit vectors of shape ``(n, 3)``.
        l_max (int): Maximum tensor rank.

    Returns:
        torch.Tensor: Stored components of shape ``(n, (l_max + 1)**2)``, on the
            input device and with its dtype.
    """
    if int(l_max) <= KERNEL_L_MAX:
        return _PY_IRREPS_FORWARD_MODULE.py_irreps_forward(x, l_max)
    return compile_py_irreps_forward(l_max)(x)


if torch.cuda.is_available() and TRITON_AVAILABLE:

    _kernels = load_generated(_GENERATED_MODULE_NAME, _CODEGEN_FILE, KERNEL_L_MAX, build_triton_irreps_module_source)

    _irreps_fwd_kernel = _kernels.irreps_fwd_kernel
    _irreps_bwd_kernel = _kernels.irreps_bwd_kernel
    _irreps_dbwd_kernel = _kernels.irreps_dbwd_kernel

    def _irreps_grid(n_batch: int):
        def grid(META):
            return (triton.cdiv(n_batch, META["BLOCK_SIZE"]),)

        return grid

    @triton_op("flashcart::irreps_fwd", mutates_args={})
    def irreps_fwd_op(x: torch.Tensor, l_max: int) -> torch.Tensor:
        """Evaluate a generated Cartesian expansion operation registered with PyTorch.

        The caller supplies normalized inputs and a rank within the generated limit.

        Returns:
            torch.Tensor: Expansion components in increasing tensor rank.
        """
        x = x.contiguous()
        n_batch = x.shape[0]
        out = x.new_empty((n_batch, _output_dim(l_max)))
        launcher(_irreps_fwd_kernel)[_irreps_grid(n_batch)](
            x_ptr=x,
            out_ptr=out,
            n_batch=n_batch,
            x_stride_n=x.stride(0),
            out_stride_n=out.stride(0),
            L_MAX=l_max,
        )
        return out

    def irreps_setup_context(ctx, inputs, output):
        x, l_max = inputs
        ctx.save_for_backward(x)
        ctx.l_max = int(l_max)

    def irreps_bwd(ctx, grad_output: torch.Tensor):
        if not ctx.needs_input_grad[0]:
            return None, None
        (x,) = ctx.saved_tensors
        return torch.ops.flashcart.irreps_bwd(grad_output, x, ctx.l_max), None

    @triton_op("flashcart::irreps_bwd", mutates_args={})
    def irreps_bwd_op(
        grad_output: torch.Tensor,
        x: torch.Tensor,
        l_max: int,
    ) -> torch.Tensor:
        """Evaluate a generated Cartesian expansion operation registered with PyTorch.

        The caller supplies normalized inputs and a rank within the generated limit.

        Returns:
            torch.Tensor: Input gradients of shape ``(n, 3)``.
        """
        grad_output = grad_output.contiguous()
        x = x.contiguous()
        n_batch = x.shape[0]
        grad_x = torch.empty_like(x)
        launcher(_irreps_bwd_kernel)[_irreps_grid(n_batch)](
            go_ptr=grad_output,
            x_ptr=x,
            gx_ptr=grad_x,
            n_batch=n_batch,
            go_stride_n=grad_output.stride(0),
            x_stride_n=x.stride(0),
            gx_stride_n=grad_x.stride(0),
            L_MAX=l_max,
        )
        return grad_x

    def irreps_bwd_setup_context(ctx, inputs, output):
        grad_output, x, l_max = inputs
        if hasattr(ctx, "set_materialize_grads"):
            ctx.set_materialize_grads(False)
        ctx.save_for_backward(grad_output, x)
        ctx.l_max = int(l_max)

    @triton_op("flashcart::irreps_dbwd", mutates_args={})
    def irreps_dbwd_op(
        grad_output: torch.Tensor,
        x: torch.Tensor,
        v: torch.Tensor,
        l_max: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Differentiate the expansion's input-gradient operation.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Derivatives with respect to the
                incoming output gradient and the input vectors, respectively.
        """
        grad_output = grad_output.contiguous()
        x = x.contiguous()
        v = v.contiguous()
        n_batch = x.shape[0]
        d_grad_out = x.new_empty((n_batch, _output_dim(l_max)))
        d_x = torch.empty_like(x)
        launcher(_irreps_dbwd_kernel)[_irreps_grid(n_batch)](
            go_ptr=grad_output,
            x_ptr=x,
            v_ptr=v,
            d_go_ptr=d_grad_out,
            d_x_ptr=d_x,
            n_batch=n_batch,
            go_stride_n=grad_output.stride(0),
            x_stride_n=x.stride(0),
            v_stride_n=v.stride(0),
            d_go_stride_n=d_grad_out.stride(0),
            d_x_stride_n=d_x.stride(0),
            L_MAX=l_max,
        )
        return d_grad_out, d_x

    def irreps_dbwd(ctx, v: torch.Tensor):
        need_d_grad_out = bool(ctx.needs_input_grad[0])
        need_d_x = bool(ctx.needs_input_grad[1])
        if v is None or not (need_d_grad_out or need_d_x):
            return None, None, None
        grad_output, x = ctx.saved_tensors
        d_grad_out, d_x = torch.ops.flashcart.irreps_dbwd(grad_output, x, v, ctx.l_max)
        return (
            d_grad_out if need_d_grad_out else None,
            d_x if need_d_x else None,
            None,
        )

    irreps_fwd_op.register_autograd(irreps_bwd, setup_context=irreps_setup_context)
    irreps_bwd_op.register_autograd(irreps_dbwd, setup_context=irreps_bwd_setup_context)

    def triton_irreps(x: "torch.Tensor", l_max: int) -> "torch.Tensor":
        """Evaluate the Cartesian expansion with the available kernel backend.

        When Triton and CUDA were available at import, CUDA inputs use the generated
        kernel and must fit its rank limit. Other calls use the PyTorch implementation.
        The input vectors must already have unit length.

        Args:
            x (torch.Tensor): Unit vectors of shape ``(n, 3)``.
            l_max (int): Maximum tensor rank.

        Returns:
            torch.Tensor: Stored components of shape ``(n, (l_max + 1)**2)``, with
                the input dtype and device.
        """
        if l_max > KERNEL_L_MAX:
            raise NotImplementedError(f"triton_irreps currently supports l_max <= {KERNEL_L_MAX}, got {l_max}.")
        if not x.is_cuda:
            warnings.warn(
                "Triton irreps kernels require CUDA tensors, but got CPU tensors. "
                "Falling back to PyTorch implementation.",
                UserWarning,
                stacklevel=2,
            )
            return py_irreps(x, l_max)
        return torch.ops.flashcart.irreps_fwd(x, l_max)

else:

    def triton_irreps(x, l_max):
        """Evaluate the Cartesian expansion with the available kernel backend.

        When Triton and CUDA were available at import, CUDA inputs use the generated
        kernel and must fit its rank limit. Other calls use the PyTorch implementation.
        The input vectors must already have unit length.

        Args:
            x (torch.Tensor): Unit vectors of shape ``(n, 3)``.
            l_max (int): Maximum tensor rank.

        Returns:
            torch.Tensor: Stored components of shape ``(n, (l_max + 1)**2)``, with
                the input dtype and device.
        """
        warnings.warn(
            "Triton irreps kernels require CUDA and an installed triton package. "
            "Falling back to PyTorch implementation.",
            UserWarning,
            stacklevel=2,
        )
        return py_irreps(x, l_max)
