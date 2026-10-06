from typing import Union

import numpy as np
import torch


def uses_periodic_shifts(pbc: object) -> bool:
    """Whether edge vectors must include periodic shift contributions.

    Args:
        pbc (object): Per-axis periodicity flags (tensor/array/None). None is treated
            as periodic.

    Returns:
        bool: Whether any periodicity flag is true, or True when the flags are
            unavailable.
    """
    if torch.is_tensor(pbc):
        return bool(torch.any(pbc).item())
    if pbc is None:
        return True
    return bool(np.any(pbc))


def get_edge_vectors(
    positions: torch.Tensor,
    cell: torch.Tensor,
    shifts: torch.Tensor,
    edge_index: torch.Tensor,
    batch: torch.Tensor,
    use_shifts: bool = True,
) -> torch.Tensor:
    """Compute receiver-to-sender edge vectors with optional periodic translations.

    Args:
        positions (torch.Tensor): Atom positions [n_atoms, 3].
        cell (torch.Tensor): Cells [n_graphs, 3, 3].
        shifts (torch.Tensor): Integer unit-cell shifts per edge [n_edges, 3].
        edge_index (torch.Tensor): Shape ``(2, n_edges)``. Row 0 contains receivers and
            row 1 contains senders.
        batch (torch.Tensor): Graph index per atom.
        use_shifts (bool, optional): Include periodic translations in the edge vectors.
            Set False for fully non-periodic batches. Default: True.

    Returns:
        torch.Tensor: Receiver-to-sender vectors with shape ``(n_edges, 3)``, including
            the specified periodic translations when ``use_shifts`` is True.
    """
    idx_i, idx_j = edge_index[0], edge_index[1]
    if not use_shifts:
        return positions[idx_j] - positions[idx_i]
    shift_vec = (shifts.to(cell.dtype).unsqueeze(-2) @ cell[batch[idx_i]]).squeeze(-2)
    return positions[idx_j] - positions[idx_i] + shift_vec


def to_voigt6(t: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Symmetrize a 3x3 tensor into Voigt order (xx, yy, zz, yz, xz, xy).

    Args:
        t (np.ndarray | torch.Tensor): Tensor(s) of shape [..., 3, 3].

    Returns:
        np.ndarray | torch.Tensor: Symmetrized components with shape ``(..., 6)``,
            preserving the input array type. Off-diagonal components are averaged,
            without an additional engineering-shear factor.
    """
    components = [
        t[..., 0, 0],
        t[..., 1, 1],
        t[..., 2, 2],
        (t[..., 1, 2] + t[..., 2, 1]) / 2,
        (t[..., 0, 2] + t[..., 2, 0]) / 2,
        (t[..., 0, 1] + t[..., 1, 0]) / 2,
    ]
    if isinstance(t, torch.Tensor):
        return torch.stack(components, dim=-1)
    return np.stack(components, axis=-1)
