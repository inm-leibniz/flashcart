"""Expand direction vectors into irreducible Cartesian tensors."""

import torch
import torch.nn as nn

from flashcart.o3._irreps import KERNEL_L_MAX as _IRREPS_KERNEL_L_MAX, py_irreps, triton_irreps


class Irreps(nn.Module):
    """Expand direction vectors into irreducible Cartesian tensors.

    For each unit vector, the expansion contains the independent components of symmetric
    traceless tensors of ranks zero through ``l_max``. The rank-l tensor contributes
    ``2 * l + 1`` components, giving ``(l_max + 1) ** 2`` components in total.

    Args:
        l_max (int): Maximum tensor rank in the expansion.
        use_triton (bool, optional): Use Triton kernels for CUDA inputs when available.
            Enabling this option requires ``l_max`` to fit within the generated rank
            limit, even when the input is on CPU. Default: False.

    Raises:
        RuntimeError: ``use_triton=True`` and ``l_max`` exceeds the generated rank
            limit configured before import.
    """

    def __init__(self, l_max: int, use_triton: bool = False):
        super().__init__()

        if use_triton and l_max > _IRREPS_KERNEL_L_MAX:
            raise RuntimeError(
                f"Irreps(use_triton=True) supports l_max <= "
                f"{_IRREPS_KERNEL_L_MAX}, got {l_max=}. Set FLASHCART_KERNEL_L_MAX "
                f"before import (see flashcart.o3.kernel_config), or use_triton=False."
            )

        self.l_max = l_max
        self.use_triton = use_triton

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize direction vectors and evaluate their Cartesian expansion.

        Args:
            x (torch.Tensor): Nonzero direction vectors of shape ``(n, 3)``. Vectors
                that remain nonunit after normalization are rejected.

        Returns:
            torch.Tensor: Independent tensor components of shape
                ``(n, (l_max + 1) ** 2)``, concatenated in increasing tensor rank.
        """
        x = torch.nn.functional.normalize(x, dim=-1)
        return self.forward_normalized(x)

    def forward_normalized(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the Cartesian expansion of unit direction vectors.

        Args:
            x (torch.Tensor): Unit vectors of shape ``(n, 3)``. The squared norm of
                every vector must differ from one by at most ``1e-5``.

        Returns:
            torch.Tensor: Independent tensor components of shape
                ``(n, (l_max + 1) ** 2)``, concatenated in increasing tensor rank.
        """
        r2 = x.pow(2).sum(dim=-1)
        torch._assert_async(
            ((r2 - 1.0).abs() <= 1.0e-5).all(),
            "Irreps.forward_normalized requires unit-length input vectors.",
        )
        if self.use_triton and x.is_cuda:
            return triton_irreps(x, self.l_max)
        return py_irreps(x, self.l_max)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"l_max={self.l_max}, "
            f"use_triton={self.use_triton}, "
            f"output_dim={(self.l_max + 1) ** 2})"
        )
