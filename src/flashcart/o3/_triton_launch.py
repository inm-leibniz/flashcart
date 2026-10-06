"""Select Triton launchers for eager execution and PyTorch dispatch contexts."""

import torch
from torch.library import wrap_triton

try:
    from torch.utils._python_dispatch import _get_current_dispatch_mode
except ImportError:
    _get_current_dispatch_mode = None


def launcher(kernel):
    """Select a Triton launcher compatible with the current dispatch context.

    Return the kernel directly when no dispatch mode is active and PyTorch is not
    compiling. Otherwise, use ``wrap_triton``. If dispatch-mode inspection is
    unavailable, use ``wrap_triton``.

    Args:
        kernel: Triton JIT kernel or autotuner to launch.

    Returns:
        object: Kernel or wrapped launcher for the current context.
    """
    if _get_current_dispatch_mode is None:
        return wrap_triton(kernel)
    if _get_current_dispatch_mode() is not None or torch.compiler.is_compiling():
        return wrap_triton(kernel)
    return kernel
