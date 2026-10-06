"""Exchange ghost-atom features and gradients through LAMMPS ML-IAP.

Forward and reverse exchanges are registered as PyTorch custom operators so they can
run inside compiled predictions.
:func:`lammps_data_slot` provides the ML-IAP communication object during evaluation.
"""

from contextlib import contextmanager
from functools import update_wrapper
from typing import Any, Iterator, Optional

import torch

_LAMMPS_DATA: Optional[Any] = None


@contextmanager
def lammps_data_slot(lammps_data: Any) -> Iterator[None]:
    """Expose an ML-IAP data object to the exchange operators during evaluation.

    Both forward evaluation and differentiation must run inside this context. The
    communication object is stored process-wide, and the previous object is restored
    on exit. Concurrent evaluations using different LAMMPS objects are not supported.

    Args:
        lammps_data (Any): ML-IAP data object providing ``forward_exchange`` and
            ``reverse_exchange``.
    """
    global _LAMMPS_DATA
    previous = _LAMMPS_DATA
    _LAMMPS_DATA = lammps_data
    try:
        yield
    finally:
        _LAMMPS_DATA = previous


def _require_lammps_data() -> Any:
    """Return the active communication object, or raise if its hooks are unavailable."""
    data = _LAMMPS_DATA
    if data is None or not hasattr(data, "forward_exchange") or not hasattr(data, "reverse_exchange"):
        raise RuntimeError(
            "LAMMPS ghost exchange requires an ML-IAP data object with forward_exchange/reverse_exchange. "
            "Evaluate the model inside flashcart.utils.lammps.lammps_data_slot(data)."
        )
    return data


def _sync_stream(tensor: torch.Tensor) -> None:
    """Synchronize the device when PyTorch uses a non-default CUDA stream.

    Called before and after each exchange. On the default stream, ordering relies
    on PyTorch using the legacy default stream and Kokkos using a blocking stream.
    Non-default streams use device synchronization to wait for both PyTorch and
    Kokkos work. CPU tensors require no synchronization.

    Args:
        tensor (torch.Tensor): Exchange tensor whose device selects the CUDA streams.
    """
    if not tensor.is_cuda:
        return
    device = tensor.device
    if torch.cuda.current_stream(device) != torch.cuda.default_stream(device):
        torch.cuda.synchronize(device)


def lammps_forward_exchange(feats: torch.Tensor) -> torch.Tensor:
    """Copy features from owning ranks to ghost atoms.

    Requires an active :func:`lammps_data_slot` context.

    Args:
        feats (torch.Tensor): Features with shape ``(ntotal, C)``, where ``ntotal``
            counts owned and ghost atoms and ``C`` is the number of features. Owned
            rows come first. Noncontiguous inputs are accepted. The input is not
            modified.

    Returns:
        torch.Tensor: Newly allocated contiguous features with the same shape, dtype,
            and device as ``feats``. Owned rows are preserved, and ghost rows hold
            the features of their owners.

    Raises:
        RuntimeError: No ML-IAP data object with both exchange hooks is active.
    """
    data = _require_lammps_data()
    feats = feats.contiguous()
    out = torch.empty_like(feats)
    _sync_stream(feats)
    data.forward_exchange(feats, out, feats.shape[-1])
    _sync_stream(feats)
    return out


def lammps_reverse_exchange(grads: torch.Tensor) -> torch.Tensor:
    """Accumulate ghost-row gradients on their owning ranks.

    Requires an active :func:`lammps_data_slot` context.

    Args:
        grads (torch.Tensor): Gradients with shape ``(ntotal, C)``, where ``ntotal``
            counts owned and ghost atoms and ``C`` is the number of features. Owned
            rows come first. Noncontiguous inputs are accepted. The input is not
            modified.

    Returns:
        torch.Tensor: Newly allocated contiguous gradients with the same shape,
            dtype, and device as ``grads``. Ghost rows are zero and their contributions
            are added to the owner rows.

    Raises:
        RuntimeError: No ML-IAP data object with both exchange hooks is active.
    """
    data = _require_lammps_data()
    grads = grads.contiguous()
    out = torch.empty_like(grads)
    _sync_stream(grads)
    data.reverse_exchange(grads, out, grads.shape[-1])
    _sync_stream(grads)
    return out


lammps_forward_exchange = update_wrapper(
    torch.library.custom_op("flashcart::lammps_forward_exchange", mutates_args=())(lammps_forward_exchange),
    lammps_forward_exchange,
)
lammps_reverse_exchange = update_wrapper(
    torch.library.custom_op("flashcart::lammps_reverse_exchange", mutates_args=())(lammps_reverse_exchange),
    lammps_reverse_exchange,
)


@lammps_forward_exchange.register_fake
def _forward_exchange_fake(feats: torch.Tensor) -> torch.Tensor:
    """Describe the forward exchange output without communicating between ranks."""
    return feats.new_empty(feats.shape)


@lammps_reverse_exchange.register_fake
def _reverse_exchange_fake(grads: torch.Tensor) -> torch.Tensor:
    """Describe the reverse exchange output without communicating between ranks."""
    return grads.new_empty(grads.shape)


def _forward_exchange_backward(ctx, grad_out: torch.Tensor) -> torch.Tensor:
    """Return input gradients by accumulating contributions from ghost atoms."""
    return torch.ops.flashcart.lammps_reverse_exchange(grad_out)


def _reverse_exchange_backward(ctx, grad_out: torch.Tensor) -> torch.Tensor:
    """Return input gradients by copying owner contributions to ghost atoms."""
    return torch.ops.flashcart.lammps_forward_exchange(grad_out)


lammps_forward_exchange.register_autograd(_forward_exchange_backward)
lammps_reverse_exchange.register_autograd(_reverse_exchange_backward)


def lammps_exchange_features(
    feats: torch.Tensor,
    nlocal: int,
    ntotal: int,
) -> torch.Tensor:
    """Refresh ghost-atom features from their owning ranks between layers.

    Features of shape ``(nlocal, C)`` are padded with zeros to ``(ntotal, C)`` before
    the exchange fills the ghost rows. All ranks take part, including ranks without
    ghost atoms, because their owned atoms may be ghosts on neighboring ranks.
    Evaluation and differentiation must run inside :func:`lammps_data_slot`.

    During differentiation, reverse communication accumulates gradients from ghost rows
    onto their owners.

    Args:
        feats (torch.Tensor): Node features with shape ``(nlocal, C)`` or
            ``(ntotal, C)``, where ``C`` is the number of features. Owned rows come
            first.
        nlocal (int): Number of atoms owned by this rank.
        ntotal (int): Total number of owned and ghost atoms on this rank.

    Returns:
        torch.Tensor: Features with shape ``(ntotal, C)`` after ghost rows have been
            filled.

    Raises:
        RuntimeError: The feature row count is incompatible with ``nlocal`` and
            ``ntotal``, or no communication object with both exchange hooks is active.
    """
    if feats.shape[0] == nlocal and ntotal > nlocal:
        pad_shape = (ntotal - nlocal, *feats.shape[1:])
        feats = torch.cat([feats, feats.new_zeros(pad_shape)], dim=0)
    elif feats.shape[0] != ntotal:
        raise RuntimeError(f"LAMMPS feature exchange expected {nlocal} or {ntotal} rows, got {feats.shape[0]}.")
    return torch.ops.flashcart.lammps_forward_exchange(feats)
