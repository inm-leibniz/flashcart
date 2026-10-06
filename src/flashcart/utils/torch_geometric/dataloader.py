"""
Trimmed-down pytorch_geometric: https://github.com/pyg-team/pytorch_geometric.
Copyright (c) 2023 PyG Team, MIT License (see NOTICE).

Local FlashCart modifications are marked with "# FlashCart:" comments.
"""

from collections.abc import Mapping, Sequence
from typing import List, Optional

import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate

# FlashCart: repo-internal import for the Collater modification below.
from flashcart.utils.geometry import uses_periodic_shifts

from .batch import Batch
from .data import Data
from .dataset import Dataset


class Collater:
    """Combine graph objects or nested values into a batch.

    Graph batches receive a ``use_shifts`` flag from their periodicity flags.

    Args:
        follow_batch (list[str]): Attributes needing additional assignment vectors.
        exclude_keys (list[str]): Graph attributes omitted from the batch.
    """

    def __init__(
        self,
        follow_batch,
        exclude_keys,
    ):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch):
        """Combine a nonempty list of examples using the type of its first element.

        Args:
            batch (list): Graphs, tensors, numbers, strings, or nested containers
                to combine.

        Returns:
            Any: Combined graph batch, tensor, list, or container of collated values.

        Raises:
            TypeError: The examples have an unsupported type.
        """
        elem = batch[0]
        if isinstance(elem, Data):
            out = Batch.from_data_list(
                batch,
                follow_batch=self.follow_batch,
                exclude_keys=self.exclude_keys,
            )
            # FlashCart: precompute the periodic-shifts flag once per batch.
            out.use_shifts = uses_periodic_shifts(getattr(out, "pbc", None))
            return out
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, "_fields"):
            return type(elem)(*(self(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f"DataLoader found invalid type: {type(elem)}")


class DataLoader(torch.utils.data.DataLoader):
    """Batch graph data and precompute the periodic-shift flag.

    Graph objects are combined with ``Batch.from_data_list``. The collater sets
    ``use_shifts`` from the batch's periodicity flags. A provided ``collate_fn`` is
    ignored because this loader installs its own collater.

    Args:
        dataset (Dataset): Dataset to load.
        batch_size (int, optional): Number of examples per batch. Used when
            ``batch_sampler`` is not provided. Default: 1.
        shuffle (bool, optional): Whether to shuffle examples. Used when
            ``batch_sampler`` is not provided. Default: False.
        follow_batch (list[str], optional): Attribute names for which to construct
            additional batch-assignment vectors. Default: [None], which selects no
            attributes.
        exclude_keys (list[str], optional): Attribute names to omit from batches.
            Default: [None], which excludes no named attributes.
        **kwargs: Additional arguments for ``torch.utils.data.DataLoader``, including an
            optional ``batch_sampler``.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = [None],
        exclude_keys: Optional[List[str]] = [None],
        **kwargs,
    ):

        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        collater = Collater(
            follow_batch,
            exclude_keys,
        )
        # FlashCart: support batch_sampler (mutually exclusive with
        # batch_size/shuffle in torch), needed for flashcart.data.samplers.DynamicBatchSampler.
        if "batch_sampler" in kwargs:
            super().__init__(dataset, collate_fn=collater, **kwargs)
        else:
            super().__init__(
                dataset,
                batch_size,
                shuffle,
                collate_fn=collater,
                **kwargs,
            )
