"""
Trimmed-down pytorch_scatter: https://github.com/rusty1s/pytorch_scatter.
Copyright (c) 2020 Matthias Fey, MIT License (see NOTICE).
"""

from typing import Optional

import torch


def _broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    """Expand bin indices to the shape of the source values for scatter addition.

    Args:
        src (torch.Tensor): Bin indices, with one-dimensional indices aligned to dim.
        other (torch.Tensor): Source values whose shape the indices must match.
        dim (int): Scatter dimension, accepting negative indices.

    Returns:
        torch.Tensor: Broadcast view of the indices with the same shape as other.
    """
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """Sum ``src`` entries with equal ``index`` along ``dim`` (scatter-add).

    Args:
        src (torch.Tensor): Values to reduce.
        index (torch.Tensor): Target bin per entry, broadcast to the shape of ``src``.
        dim (int, optional): Dimension to reduce along. Default: -1.
        out (torch.Tensor, optional): Tensor into which values are added in place.
            Default: None, which allocates a zero-initialized output.
        dim_size (int, optional): Output size along ``dim`` when allocating an output.
            Default: None, which uses the largest index plus one, or zero for an empty
            index tensor.

    Returns:
        torch.Tensor: Sum for each output index. If ``out`` is provided, returns that
            tensor after accumulation. Otherwise, returns the newly allocated tensor.
    """
    index = _broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)
