from collections.abc import Sized
from typing import Iterator, List, Optional

import torch


class DistributedStridedSampler(torch.utils.data.Sampler):
    """Assign each dataset index to one distributed rank.

    Rank ``r`` receives ``indices[r::n_replicas]`` after optional shuffling. Samples are
    not duplicated to equalize rank lengths, so the numbers of samples and batches may
    differ across ranks. This supports evaluation that reduces accumulated statistics
    after iteration. Training with synchronized gradient reductions requires compatible
    iteration counts.

    Args:
        dataset (Sized): Dataset providing its length.
        n_replicas (int): World size.
        rank (int): Rank of this process.
        shuffle (bool, optional): Reshuffle globally each epoch. Default: False.
        seed (int, optional): Base seed. The epoch is added to it. Default: 0.
    """

    def __init__(
        self,
        dataset: Sized,
        n_replicas: int,
        rank: int,
        shuffle: bool = False,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.n_replicas = max(int(n_replicas), 1)
        self.rank = int(rank)
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the shuffle epoch (called by the training loop).

        Args:
            epoch (int): Epoch number mixed into the shuffle seed.
        """
        self._epoch = epoch

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self._epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        return iter(indices[self.rank :: self.n_replicas])

    def __len__(self) -> int:
        n = len(self.dataset)
        return len(range(self.rank, n, self.n_replicas))


class DynamicBatchSampler(torch.utils.data.Sampler):
    """Group variable-size graphs within atom and edge budgets.

    Indices are optionally shuffled and then distributed by rank. Each rank adds graphs
    in order until the next graph would exceed a budget. A graph that exceeds a budget
    forms a batch by itself.

    With distributed ``balance_batches=True``, all ranks use the same number of
    batches. ``drop_last=True`` reduces this to the smallest batch count. Otherwise,
    ranks with fewer batches repeat their final batch. A rank with no samples uses a
    sample from the full index list when needed. Balanced sampling can therefore
    discard or repeat samples.

    The shuffle epoch advances after a complete iteration. If iteration stops early,
    the shuffle epoch does not advance.

    Args:
        sizes (list[tuple[int, int]]): (n_nodes, n_edges) per structure.
        max_nodes (int): Node budget per batch.
        max_edges (int): Edge budget per batch.
        shuffle (bool, optional): Reshuffle each epoch. Default: True.
        seed (int, optional): Base seed. The epoch is added to it. Default: 0.
        n_replicas (int, optional): World size. Default: 1.
        rank (int, optional): Rank of this process. Default: 0.
        drop_last (bool, optional): Truncate balanced distributed ranks to the smallest
            batch count. Has no effect without distributed batch balancing. Default:
            False.
        balance_batches (bool, optional): Equal batch counts across ranks. Default:
            False.
    """

    def __init__(
        self,
        sizes: List[tuple[int, int]],
        max_nodes: int,
        max_edges: int,
        shuffle: bool = True,
        seed: int = 0,
        n_replicas: int = 1,
        rank: int = 0,
        drop_last: bool = False,
        balance_batches: bool = False,
    ) -> None:
        self.sizes = sizes
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.shuffle = shuffle
        self.seed = seed
        self.n_replicas = max(int(n_replicas), 1)
        self.rank = int(rank)
        self.drop_last = drop_last
        self.balance_batches = balance_batches
        self._epoch = 0
        self._max_graphs: Optional[int] = None

    @property
    def max_graphs(self) -> int:
        """Upper bound on structures per batch, over every possible shuffle.

        Node and edge counts are sorted separately. Adding the smallest counts first
        can fit at least as many structures as any actual batch, so the result is an
        upper bound. A structure that exceeds a budget can still form its own batch.

        Returns:
            int: Upper bound on the number of graphs in a batch, with a minimum of one,
                including for an empty dataset.
        """
        if self._max_graphs is None:
            nodes = sorted(n for n, _ in self.sizes)
            edges = sorted(e for _, e in self.sizes)
            count = total_nodes = total_edges = 0
            for n, e in zip(nodes, edges):
                if count and (total_nodes + n > self.max_nodes or total_edges + e > self.max_edges):
                    break
                total_nodes += n
                total_edges += e
                count += 1
            self._max_graphs = max(count, 1)
        return self._max_graphs

    @classmethod
    def from_target_batch_size(
        cls,
        sizes: List[tuple[int, int]],
        target_batch_size: int,
        factor: float = 1.1,
        shuffle: bool = True,
        seed: int = 0,
        n_replicas: int = 1,
        rank: int = 0,
        drop_last: bool = False,
        balance_batches: bool = False,
    ) -> "DynamicBatchSampler":
        """Derive atom and edge budgets from a target batch size.

        Budgets are computed separately as ``factor * target_batch_size * mean_size``,
        converted to integers with a minimum of one. ``sizes`` must be nonempty. The
        target controls the budgets. Actual batch sizes depend on graph sizes and their
        order.

        Args:
            sizes (list[tuple[int, int]]): (n_nodes, n_edges) per structure.
            target_batch_size (int): Average structures per batch to aim for.
            factor (float, optional): Multiplier for the budgets computed from average
                graph sizes. Default: 1.1.
            shuffle (bool, optional): Reshuffle each epoch. Default: True.
            seed (int, optional): Base seed. Default: 0.
            n_replicas (int, optional): World size. Default: 1.
            rank (int, optional): Rank of this process. Default: 0.
            drop_last (bool, optional): Truncate balanced distributed ranks to the
                smallest batch count. Has no effect without distributed batch balancing.
                Default: False.
            balance_batches (bool, optional): Equal batch counts across ranks. Default:
                False.

        Returns:
            DynamicBatchSampler: Sampler initialized with the derived budgets.
        """
        avg_nodes = sum(n for n, e in sizes) / len(sizes)
        avg_edges = sum(e for n, e in sizes) / len(sizes)
        max_nodes = max(1, int(factor * target_batch_size * avg_nodes))
        max_edges = max(1, int(factor * target_batch_size * avg_edges))
        return cls(
            sizes,
            max_nodes=max_nodes,
            max_edges=max_edges,
            shuffle=shuffle,
            seed=seed,
            n_replicas=n_replicas,
            rank=rank,
            drop_last=drop_last,
            balance_batches=balance_batches,
        )

    def set_epoch(self, epoch: int) -> None:
        """Set the shuffle epoch (called by the training loop).

        Args:
            epoch (int): Epoch number mixed into the shuffle seed.
        """
        self._epoch = epoch

    # The epoch advances only after a completed pass: an abandoned partial
    # iteration replays the same epoch's batches.
    def __iter__(self) -> Iterator[List[int]]:
        for batch in self._balanced_batches():
            yield batch
        self._epoch += 1

    def _global_indices(self) -> List[int]:
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self._epoch)
            return torch.randperm(len(self.sizes), generator=g).tolist()
        return list(range(len(self.sizes)))

    def _rank_indices(self, rank: int) -> List[int]:
        return self._global_indices()[rank :: self.n_replicas]

    # Start a new batch when adding the next graph would exceed a budget.
    # A graph larger than a budget still gets its own batch.
    def _make_batches(self, indices: List[int]) -> List[List[int]]:
        batches: List[List[int]] = []
        batch: List[int] = []
        total_nodes = total_edges = 0
        for idx in indices:
            n, e = self.sizes[idx]
            if batch and (total_nodes + n > self.max_nodes or total_edges + e > self.max_edges):
                batches.append(batch)
                batch = []
                total_nodes = total_edges = 0
            batch.append(idx)
            total_nodes += n
            total_edges += e
        if batch:
            batches.append(batch)
        return batches

    def _rank_batches(self, rank: int) -> List[List[int]]:
        return self._make_batches(self._rank_indices(rank))

    # The shared seed lets each rank compute all batch counts without communication.
    # Ranks above the target discard batches. Ranks below it repeat their last batch.
    # A rank with no batches uses one sample from the full index list.
    # Equal batch counts give every rank the same number of gradient reductions.
    def _balanced_batches(self) -> List[List[int]]:
        batches = self._rank_batches(self.rank)
        if self.n_replicas == 1 or not self.balance_batches:
            return batches

        lengths = [len(self._rank_batches(rank)) for rank in range(self.n_replicas)]
        target = min(lengths) if self.drop_last else max(lengths)
        if len(batches) >= target:
            return batches[:target]
        if not batches:
            indices = self._global_indices()
            return [[indices[self.rank % len(indices)]]] * target if indices else []
        return batches + [batches[-1]] * (target - len(batches))

    def __len__(self) -> int:
        return len(self._balanced_batches())
