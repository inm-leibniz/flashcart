from pathlib import Path
from typing import List, Union

import ase.data
import ase.io
import numpy as np

__all__ = [
    "numbers_to_types",
    "types_to_one_hot",
    "get_elements",
    "read_extxyz",
    "write_extxyz",
    "read_hdf5",
    "write_hdf5",
]


def numbers_to_types(
    atomic_numbers: np.ndarray,
    elements: List[Union[str, int]],
) -> np.ndarray:
    """Map atomic numbers to 0-based indices into ``elements``.

    The returned indices are the model's internal atom types. Their order is fixed by
    ``elements`` (the training element order).

    Args:
        atomic_numbers (np.ndarray): Per-atom atomic numbers.
        elements (list[str | int]): Element symbols or atomic numbers. Raises on
            duplicates or on atomic numbers not covered by the list.

    Returns:
        np.ndarray: Zero-based element indices with the same shape as ``atomic_numbers``
            and dtype ``int64``.
    """
    element_numbers = np.asarray(
        [ase.data.atomic_numbers[e] if isinstance(e, str) else int(e) for e in elements],
        dtype=np.int64,
    )
    if len(np.unique(element_numbers)) != len(element_numbers):
        raise ValueError(f"Duplicate elements are not allowed: {element_numbers.tolist()}.")
    max_z = int(element_numbers.max())
    mapping = -np.ones(max_z + 2, dtype=np.int64)
    mapping[element_numbers] = np.arange(len(element_numbers), dtype=np.int64)
    zs = np.asarray(atomic_numbers, dtype=np.int64)
    atom_types = mapping[np.minimum(zs, max_z + 1)]
    missing_mask = atom_types < 0
    if missing_mask.any():
        missing = sorted(set(zs[missing_mask].tolist()))
        raise ValueError(
            f"Configuration contains atomic numbers not in elements={element_numbers.tolist()}: " f"{missing}."
        )
    return atom_types


def types_to_one_hot(atom_types: np.ndarray, n_elements: int) -> np.ndarray:
    """One-hot encode atom types.

    Args:
        atom_types (np.ndarray): 0-based type indices.
        n_elements (int): Number of columns of the one-hot matrix.

    Returns:
        np.ndarray: One-hot matrix with shape ``(n_atoms, n_elements)`` and dtype
            ``float64``.
    """
    oh = np.zeros((len(atom_types), n_elements), dtype=np.float64)
    oh[np.arange(len(atom_types)), atom_types] = 1.0
    return oh


def get_elements(
    file_path: Union[str, Path],
    energy_key: str = "REF_energy",
    forces_key: str = "REF_forces",
) -> List[str]:
    """Read an extxyz file and list its elements in increasing atomic-number order.

    Args:
        file_path (str | Path): extxyz file to scan.
        energy_key (str, optional): Key for the reference energy in ``atoms.info``.
            Default: "REF_energy".
        forces_key (str, optional): Key for the reference forces in ``atoms.arrays``.
            Default: "REF_forces".

    Returns:
        list[str]: Element symbols ordered by increasing atomic number.
    """
    configs = read_extxyz(file_path, index=":", energy_key=energy_key, forces_key=forces_key)
    unique_z = sorted({z for c in configs for z in c.atomic_numbers})
    return [ase.data.chemical_symbols[z] for z in unique_z]


def read_extxyz(
    file_path: Union[str, Path],
    index: Union[str, int, slice] = ":",
    energy_key: str = "REF_energy",
    forces_key: str = "REF_forces",
    stress_key: str = "REF_stress",
) -> list["AtomicConfig"]:
    """Read extxyz frames into AtomicConfig objects.

    Args:
        file_path (str | Path): extxyz file.
        index (str | int | slice, optional): ase-style frame selection. Default: ":"
            (all frames).
        energy_key (str, optional): Key for the reference energy in ``atoms.info``.
            Default: "REF_energy".
        forces_key (str, optional): Key for the reference forces in ``atoms.arrays``.
            Default: "REF_forces".
        stress_key (str, optional): Key for the reference stress in ``atoms.info``.
            Default: "REF_stress".

    Returns:
        list[AtomicConfig]: Selected configurations in the requested order, including a
            one-element list when one frame is selected.
    """
    from flashcart.data.data import AtomicConfig

    atoms_list = ase.io.read(file_path, index=index, format="extxyz")
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    return [
        AtomicConfig.from_ase(a, energy_key=energy_key, forces_key=forces_key, stress_key=stress_key)
        for a in atoms_list
    ]


def write_extxyz(
    file_path: Union[str, Path],
    configs: list["AtomicConfig"],
    energy_key: str = "REF_energy",
    forces_key: str = "REF_forces",
    stress_key: str = "REF_stress",
    append: bool = False,
) -> None:
    """Write atomic configurations and their reference properties as extxyz frames.

    Args:
        file_path (str | Path): Output file.
        configs (list[AtomicConfig]): Structures to write.
        energy_key (str, optional): Key for the reference energy in ``atoms.info``.
            Default: "REF_energy".
        forces_key (str, optional): Key for the reference forces in ``atoms.arrays``.
            Default: "REF_forces".
        stress_key (str, optional): Key for the reference stress in ``atoms.info``.
            Default: "REF_stress".
        append (bool, optional): Append instead of overwrite. Default: False.
    """
    ase.io.write(
        file_path,
        [c.to_ase(energy_key=energy_key, forces_key=forces_key, stress_key=stress_key) for c in configs],
        format="extxyz",
        append=append,
    )


def write_hdf5(
    file_path: Union[str, Path],
    configs: list["AtomicConfig"],
) -> None:
    """Write atomic configurations to HDF5, replacing any existing file.

    Each frame is stored in a ``structure_%08d`` group containing its geometry and
    available reference properties. Neighbor lists are not stored.

    Args:
        file_path (str | Path): Output file.
        configs (list[AtomicConfig]): Structures to write.
    """
    import h5py

    with h5py.File(file_path, "w") as f:
        for idx, config in enumerate(configs):
            grp = f.create_group(f"structure_{idx:08d}")
            grp["atomic_numbers"] = config.atomic_numbers
            grp["positions"] = config.positions
            grp["cell"] = config.cell
            grp["pbc"] = config.pbc
            grp["energy"] = str(config.energy)  # "None" if absent
            grp["forces"] = config.forces if config.forces is not None else "None"
            grp["stress"] = config.stress if config.stress is not None else "None"


def _hdf5_decode(value) -> object:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return None if value == "None" else value


def _hdf5_group_to_config(group) -> "AtomicConfig":
    from flashcart.data.data import AtomicConfig

    return AtomicConfig(
        atomic_numbers=_hdf5_decode(group["atomic_numbers"][()]),
        positions=_hdf5_decode(group["positions"][()]),
        cell=_hdf5_decode(group["cell"][()]),
        pbc=_hdf5_decode(group["pbc"][()]),
        energy=_hdf5_decode(group["energy"][()]) if "energy" in group else None,
        forces=_hdf5_decode(group["forces"][()]) if "forces" in group else None,
        stress=_hdf5_decode(group["stress"][()]) if "stress" in group else None,
    )


def read_hdf5(file_path: Union[str, Path]) -> list["AtomicConfig"]:
    """Read all structure groups of an HDF5 file written by ``write_hdf5``.

    Args:
        file_path (str | Path): Input file.

    Returns:
        list[AtomicConfig]: Configurations read in sorted group-name order.
    """
    import h5py

    with h5py.File(file_path, "r") as f:
        return [_hdf5_group_to_config(f[key]) for key in sorted(f.keys())]
