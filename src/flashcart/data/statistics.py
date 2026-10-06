from typing import Optional, Union

import numpy as np
import torch


def get_atomic_energy_shifts(
    data_loader: "DataLoader",
    n_elements: int,
    atomic_shifts: Optional[np.ndarray] = None,
    lam: float = 1.0,
    fit: bool = True,
) -> np.ndarray:
    """Estimate per-element energy offsets by regularized least squares.

    The provided offsets define a prior. Their summed contribution is subtracted from
    each reference energy before fitting a correction. The residual energy per atom,
    averaged across all atoms, supplies a common offset. Ridge regularization is applied
    to the remaining element-specific corrections.

    Batches without energies or atom types are skipped. If no usable batches remain, the
    prior is returned. With ``fit=False``, provided offsets are returned as a float64
    array without fitting.

    Args:
        data_loader (DataLoader): Batches with reference energies and atom types.
        n_elements (int): Number of elements.
        atomic_shifts (np.ndarray, optional): Prior offsets, with shape
            ``(n_elements,)``. Required when ``fit=False``. Default: None, which uses
            zeros when fitting.
        lam (float, optional): Ridge regularization strength for the fitted corrections.
            Default: 1.0.
        fit (bool, optional): Fit residual shifts on top of the prior. Default: True.

    Returns:
        np.ndarray: Per-element energy offsets, with shape ``(n_elements,)`` and dtype
            ``float64``.
    """
    if not fit and atomic_shifts is None:
        raise ValueError("atomic_shifts must be provided when fit=False.")

    s0 = (
        torch.zeros(n_elements, dtype=torch.float64)
        if atomic_shifts is None
        else torch.as_tensor(atomic_shifts, dtype=torch.float64)
    )
    if s0.shape != (n_elements,):
        raise ValueError(f"atomic_shifts must have shape ({n_elements},). Provided: {tuple(s0.shape)}.")
    if not fit:
        return s0.numpy()

    Z_blocks: list[torch.Tensor] = []
    E_blocks: list[torch.Tensor] = []
    for batch in data_loader:
        if batch.energy is None or getattr(batch, "atom_types", None) is None:
            continue
        n_graphs = int(batch.n_atoms.shape[0])
        per_graph_counts = torch.zeros((n_graphs, n_elements), dtype=torch.float64)
        for g in range(n_graphs):
            lo, hi = int(batch.ptr[g]), int(batch.ptr[g + 1])
            per_graph_counts[g] = torch.bincount(batch.atom_types[lo:hi].cpu(), minlength=n_elements).to(torch.float64)
        E_blocks.append(batch.energy[:n_graphs].reshape(-1).to(torch.float64).cpu() - per_graph_counts @ s0)
        Z_blocks.append(per_graph_counts)

    if not Z_blocks:
        return s0.numpy()

    Z = torch.cat(Z_blocks)  # (n_structures, n_elements)
    E = torch.cat(E_blocks)  # (n_structures,)

    N = Z.sum(dim=1)
    mean_e = E.sum() / N.sum()
    E_centered = E - N * mean_e

    Z_aug = torch.cat([Z, lam**0.5 * torch.eye(n_elements, dtype=torch.float64)])
    E_aug = torch.cat([E_centered, torch.zeros(n_elements, dtype=torch.float64)])
    solution = torch.linalg.lstsq(Z_aug, E_aug, driver="gelsd").solution

    return (solution + mean_e + s0).numpy()


def get_force_rms(
    data_loader: "DataLoader",
    n_elements: Optional[int] = None,
) -> Union[np.ndarray, float]:
    """Compute the root mean square of the reference force components.

    All Cartesian components and atoms contribute equally. Batches without forces are
    skipped.

    Args:
        data_loader (DataLoader): Batches with reference forces.
        n_elements (int, optional): If given, return an array of this length with the
            same value for every element, replacing zero by 1.0. Default: None,
            which returns a scalar.

    Returns:
        float | np.ndarray: Scalar force-component RMS when ``n_elements`` is None, or a
            float64 array containing the same value for every element. Zero is replaced
            by one only in the array form. With no reference forces, the scalar is zero
            and the array contains ones.
    """
    force_sq_sum = 0.0
    atom_sum = 0
    for batch in data_loader:
        forces = getattr(batch, "forces", None)
        if forces is None:
            continue
        forces = forces.to(torch.float64)
        force_sq_sum += forces.square().sum().item()
        atom_sum += forces.shape[0]

    rms = float(np.sqrt(force_sq_sum / max(atom_sum * 3, 1)))
    if n_elements is None:
        return rms
    values = np.full(n_elements, rms, dtype=np.float64)
    values[values == 0.0] = 1.0
    return values


def get_avg_neighbors(data_loader: "DataLoader", r_max: Optional[float] = None) -> float:
    """Average neighbor count per atom over the loader.

    Args:
        data_loader (DataLoader): Batches with edge indices.
        r_max (float, optional): Recount edges within this radius (needed when the lists
            were built with a skin). Default: None (raw edge count).

    Returns:
        float: Total included edge count divided by the total atom count, or zero when
            the loader contains no atoms.
    """
    from flashcart.utils.geometry import get_edge_vectors, uses_periodic_shifts

    total_edges = 0
    total_atoms = 0
    for batch in data_loader:
        total_atoms += int(batch.positions.shape[0])
        edge_index = getattr(batch, "edge_index", None)
        if edge_index is None or edge_index.shape[1] == 0:
            continue
        if r_max is not None:
            vectors = getattr(batch, "vectors", None)
            if vectors is None:
                vectors = get_edge_vectors(
                    batch.positions,
                    batch.cell,
                    batch.shifts,
                    edge_index,
                    batch.batch,
                    use_shifts=uses_periodic_shifts(getattr(batch, "use_shifts", True)),
                )
            total_edges += int((vectors.norm(dim=-1) < r_max).sum())
        else:
            total_edges += int(edge_index.shape[1])
    if total_atoms == 0:
        return 0.0
    return total_edges / total_atoms
