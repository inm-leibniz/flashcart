from pathlib import Path
from typing import Any, Optional, Sequence, Union

import h5py
import numpy as np
import torch
from torch.utils.data import DistributedSampler, Subset

from flashcart.data.data import AtomicConfig, AtomicData
from flashcart.data.utils import _hdf5_group_to_config, numbers_to_types
from flashcart.data.samplers import DistributedStridedSampler, DynamicBatchSampler
from flashcart.utils.torch_geometric.dataloader import DataLoader
from flashcart.utils.torch_geometric.dataset import Dataset


def make_loader(
    path: Union[str, Path],
    elements: list[str],
    r_max: float,
    batch_size: int,
    energy_key: str = "REF_energy",
    forces_key: str = "REF_forces",
    shuffle: bool = False,
    dynamic_batching: bool = True,
    n_replicas: int = 1,
    rank: int = 0,
    drop_last: bool = False,
    balance_batches: bool = False,
    data_seed: int = 0,
    atomic_shifts: Optional[Sequence[float]] = None,
    n_workers: int = 0,
) -> DataLoader:
    """Construct an extxyz data loader with fixed or variable batch sizes.

    Dynamic batching groups structures within atom and edge budgets computed from
    ``batch_size``. For fixed distributed batching, ranks use ``DistributedSampler``
    when ``balance_batches`` or ``drop_last`` is set. Otherwise, rank-strided sampling
    assigns each structure to one rank without duplication.

    Args:
        path (str | Path): extxyz file.
        elements (list[str]): Element order defining atom types.
        r_max (float): Neighbor cutoff radius.
        batch_size (int): Structures per batch on this rank. With dynamic batching,
            this value determines the atom and edge budgets. The caller must divide
            a global target among ranks before calling this function.
        energy_key (str, optional): Key for the reference energy in ``atoms.info``.
            Default: "REF_energy".
        forces_key (str, optional): Key for the reference forces in ``atoms.arrays``.
            Default: "REF_forces".
        shuffle (bool, optional): Shuffle per epoch. Default: False.
        dynamic_batching (bool, optional): Group structures by atom and edge budgets.
            Default: True.
        n_replicas (int, optional): Number of distributed ranks. Default: 1.
        rank (int, optional): Index of this rank. Default: 0.
        drop_last (bool, optional): For fixed batching, discard incomplete batches. For
            dynamic distributed batching with ``balance_batches``, truncate all ranks to
            the smallest rank batch count. Otherwise, dynamic batching retains the final
            batch. Default: False.
        balance_batches (bool, optional): Equalize the number of batches across
            distributed ranks. This may repeat samples or discard batches, depending on
            the sampler and ``drop_last``. Default: False.
        data_seed (int, optional): Seed used by shuffling and sampling. Default: 0.
        atomic_shifts (Sequence[float], optional): Per-element energy offsets, ordered
            as ``elements``, whose sum is subtracted from each reference energy during
            graph construction. Default: None.
        n_workers (int, optional): DataLoader worker processes. Default: 0.

    Returns:
        DataLoader: Loader yielding batches of atomic graphs.
    """
    data_seed = int(data_seed)
    dataset = AtomicDataset.from_extxyz(
        path,
        r_max=r_max,
        elements=elements,
        energy_key=energy_key,
        forces_key=forces_key,
        atomic_shifts=atomic_shifts,
    )
    if dynamic_batching:
        sizes = dataset.get_sizes()
        sampler = DynamicBatchSampler.from_target_batch_size(
            sizes,
            target_batch_size=batch_size,
            shuffle=shuffle,
            seed=data_seed,
            n_replicas=n_replicas,
            rank=rank,
            drop_last=drop_last,
            balance_batches=balance_batches,
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=n_workers,
            persistent_workers=n_workers > 0,
        )
    sampler = None
    if n_replicas > 1:
        if balance_batches or drop_last:
            sampler = DistributedSampler(
                dataset,
                num_replicas=n_replicas,  # torch API name
                rank=rank,
                shuffle=shuffle,
                drop_last=drop_last,
                seed=data_seed,
            )
        else:
            sampler = DistributedStridedSampler(
                dataset,
                n_replicas=n_replicas,
                rank=rank,
                shuffle=shuffle,
                seed=data_seed,
            )
        shuffle = False
    generator = torch.Generator()
    generator.manual_seed(data_seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=drop_last,
        num_workers=n_workers,
        persistent_workers=n_workers > 0,
        generator=generator,
    )


class AtomicDataset(Dataset):
    """In-memory dataset of AtomicConfig structures.

    Neighbor lists are built when needed and stored on each configuration for reuse.
    When ``atomic_shifts`` is set, the summed per-element shifts are subtracted from
    each structure's reference energy when its graph is built. The model then learns
    energies relative to these offsets. Calculators can add the shifts back.

    Args:
        configs (list[AtomicConfig]): Structures.
        r_max (float): Neighbor cutoff radius.
        elements (list[str | int]): Element order defining atom types.
        skin (float, optional): Extra neighbor radius. Default: 0.0.
        atomic_shifts (Sequence[float], optional): Per-element energy offsets, ordered
            as ``elements``, whose sum is subtracted from each reference energy during
            graph construction. Default: None.
    """

    def __init__(
        self,
        configs: list[AtomicConfig],
        r_max: float,
        elements: list[Union[str, int]],
        skin: float = 0.0,
        atomic_shifts: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        self.configs = list(configs)
        self.r_max = r_max
        self.elements = list(elements)
        self.skin = skin
        self.atomic_shifts = None if atomic_shifts is None else np.asarray(atomic_shifts, dtype=np.float64)

    @classmethod
    def from_extxyz(
        cls,
        file_path: Union[str, Path],
        r_max: float,
        elements: list[Union[str, int]],
        skin: float = 0.0,
        index: Union[str, int, slice] = ":",
        energy_key: str = "REF_energy",
        forces_key: str = "REF_forces",
        stress_key: str = "REF_stress",
        atomic_shifts: Optional[Sequence[float]] = None,
    ) -> "AtomicDataset":
        """Read selected extxyz frames into an atomic dataset.

        Args:
            file_path (str | Path): Input extxyz file.
            r_max (float): Neighbor-list cutoff.
            elements (list[str | int]): Element order defining atom types.
            skin (float, optional): Additional neighbor-list radius. Default: 0.0.
            index (str | int | slice, optional): ASE frame selection. Default: ":",
                which selects all frames.
            energy_key (str, optional): Key for the reference energy in ``atoms.info``.
                Default: "REF_energy".
            forces_key (str, optional): Key for the reference forces in
                ``atoms.arrays``. Default: "REF_forces".
            stress_key (str, optional): Key for the reference stress in ``atoms.info``.
                Default: "REF_stress".
            atomic_shifts (Sequence[float], optional): Per-element energy offsets,
                ordered as ``elements``, whose sum is subtracted from each reference
                energy during graph construction. Default: None.

        Returns:
            AtomicDataset: Dataset containing the selected configurations.
        """
        from flashcart.data.utils import read_extxyz

        configs = read_extxyz(
            file_path,
            index=index,
            energy_key=energy_key,
            forces_key=forces_key,
            stress_key=stress_key,
        )
        return cls(configs, r_max=r_max, elements=elements, skin=skin, atomic_shifts=atomic_shifts)

    def len(self) -> int:
        return len(self.configs)

    # Sum the per-element shifts to subtract from one structure's energy.
    def _energy_offset(self, config: AtomicConfig) -> float:
        if self.atomic_shifts is None:
            return 0.0
        atom_types = numbers_to_types(config.atomic_numbers, self.elements)
        return float(self.atomic_shifts[atom_types].sum())

    def get(self, idx: int) -> AtomicData:
        """Construct the graph for one configuration.

        The neighbor list is built or reused, and the summed per-element energy offset
        is subtracted when offsets have been provided.

        Args:
            idx (int): Configuration index.

        Returns:
            AtomicData: Graph containing the configuration and its reference properties.
        """
        config = self.configs[idx]
        config.compute_neighbors(self.r_max, skin=self.skin)
        return AtomicData.from_config(config, self.elements, energy_offset=self._energy_offset(config))

    def get_sizes(self) -> list[tuple[int, int]]:
        """Count atoms and neighbor-list edges in each configuration.

        Neighbor lists are built or reused before their edges are counted.

        Returns:
            list[tuple[int, int]]: Atom and edge counts for each configuration, in
                dataset order.
        """
        sizes = []
        for config in self.configs:
            config.compute_neighbors(self.r_max, skin=self.skin)
            n_nodes = len(config.atomic_numbers)
            n_edges = config.neighbors.edge_index.shape[1]
            sizes.append((n_nodes, n_edges))
        return sizes


class HDF5Dataset(Dataset):
    """Read atomic configurations from HDF5 when they are requested.

    The file is opened on first access, and the open handle is excluded when the dataset
    is pickled. Each requested configuration is read and converted to a graph with a
    newly constructed neighbor list.

    Args:
        file_path (str | Path): HDF5 file written by ``write_hdf5``.
        r_max (float): Neighbor cutoff radius.
        elements (list[str | int]): Element order defining atom types.
        skin (float, optional): Extra neighbor radius. Default: 0.0.
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        r_max: float,
        elements: list[Union[str, int]],
        skin: float = 0.0,
    ):
        super().__init__()
        self.file_path = Path(file_path)
        self.r_max = r_max
        self.elements = list(elements)
        self.skin = skin
        self._file: Optional[h5py.File] = None
        self._keys: Optional[list[str]] = None

    @property
    def _h5(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.file_path, "r")
        return self._file

    @property
    def _sorted_keys(self) -> list[str]:
        if self._keys is None:
            self._keys = sorted(self._h5.keys())
        return self._keys

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def len(self) -> int:
        return len(self._sorted_keys)

    def get(self, idx: int) -> AtomicData:
        """Read one configuration and construct its atomic graph.

        Args:
            idx (int): Index in the sorted HDF5 group names.

        Returns:
            AtomicData: Graph containing the stored reference properties and a neighbor
                list constructed at ``r_max + skin``.
        """
        config = _hdf5_group_to_config(self._h5[self._sorted_keys[idx]])
        config.compute_neighbors(self.r_max, skin=self.skin)
        return AtomicData.from_config(config, self.elements)


def random_split_indices(
    n_samples: int,
    sizes: dict[str, int],
    seed: Optional[int] = None,
) -> dict[str, np.ndarray]:
    """Partition a random permutation of sample indices into named splits.

    Splits are filled in the order given by ``sizes``. Any remaining indices go to
    ``"test"``. Sizes should be nonnegative and sum to at most ``n_samples``. If a
    remainder is expected, reserve the name ``"test"`` for it. An existing entry with
    that name is overwritten.

    Args:
        n_samples (int): Total number of samples.
        sizes (dict[str, int]): Number of samples for each split, in insertion order.
        seed (int, optional): Random-number generator seed. Default: None.

    Returns:
        dict[str, np.ndarray]: Sample indices for each named split.
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(np.arange(n_samples))
    result: dict[str, np.ndarray] = {}
    cursor = 0
    for name, size in sizes.items():
        result[name] = indices[cursor : cursor + size]
        cursor += size
    if cursor < n_samples:
        result["test"] = indices[cursor:]
    return result


def split_dataset(
    dataset,
    sizes: dict[str, int],
    seed: Optional[int] = None,
) -> dict[str, Subset]:
    """Split a dataset into subsets using ``random_split_indices``.

    Split sizes and the reserved ``"test"`` remainder follow ``random_split_indices``.

    Args:
        dataset: Dataset to split.
        sizes (dict[str, int]): Number of samples for each split, in insertion order.
        seed (int, optional): Random-number generator seed. Default: None.

    Returns:
        dict[str, Subset]: Dataset subsets keyed by split name.
    """
    return {
        name: Subset(dataset, indices.tolist())
        for name, indices in random_split_indices(len(dataset), sizes, seed).items()
    }
