"""
Trimmed-down pytorch_geometric: https://github.com/pyg-team/pytorch_geometric.
Copyright (c) 2023 PyG Team, MIT License (see NOTICE).

Local FlashCart modifications are marked with "# FlashCart:" comments.
"""

import torch
from torch import Tensor

from .data import Data


class Batch(Data):
    """Represent a batch of graphs as one disconnected graph.

    The ``batch`` vector assigns each node to its graph. The ``ptr`` vector stores
    cumulative node counts. Tensor attributes are concatenated using the rules defined
    by ``Data``.

    Args:
        batch (torch.Tensor, optional): Graph index for each node. Default: None.
        ptr (torch.Tensor, optional): Cumulative node counts, including an initial zero.
            Default: None.
        **kwargs: Graph attributes passed to ``Data``.
    """

    def __init__(self, batch=None, ptr=None, **kwargs):
        super(Batch, self).__init__(**kwargs)

        for key, item in kwargs.items():
            if key == "num_nodes":
                self.__num_nodes__ = item
            else:
                self[key] = item

        self.batch = batch
        self.ptr = ptr
        self.__data_class__ = Data

    @classmethod
    def from_data_list(cls, data_list, follow_batch=[], exclude_keys=[]):
        """Construct a batch from a nonempty list of graph objects.

        Attributes are concatenated using each graph's batching rules. Index attributes
        are offset to refer to nodes in the combined graph.

        Args:
            data_list (list[Data]): Graphs with compatible attributes and layouts.
            follow_batch (list[str], optional): Attribute names for which to construct
                additional assignment vectors. Default: [].
            exclude_keys (list[str], optional): Attribute names omitted from the batch.
                Default: [].

        Returns:
            Batch: Combined graph with node assignments and cumulative node counts.
        """
        keys = list(set(data_list[0].keys) - set(exclude_keys))
        assert "batch" not in keys and "ptr" not in keys

        batch = cls()
        for key in data_list[0].__dict__.keys():
            if key[:2] != "__" and key[-2:] != "__":
                batch[key] = None

        batch.__data_class__ = data_list[0].__class__
        for key in keys + ["batch"]:
            batch[key] = []
        batch["ptr"] = [0]

        device = None
        cumsum = {key: [0] for key in keys}
        for i, data in enumerate(data_list):
            for key in keys:
                item = data[key]

                # Increase values by `cumsum` value.
                cum = cumsum[key][-1]
                if isinstance(item, Tensor) and item.dtype != torch.bool:
                    if not isinstance(cum, int) or cum != 0:
                        item = item + cum
                elif isinstance(item, (int, float)):
                    item = item + cum

                # Gather the size of the `cat` dimension.
                size = 1
                cat_dim = data.__cat_dim__(key, data[key])
                # 0-dimensional tensors have no dimension along which to
                # concatenate, so we set `cat_dim` to `None`.
                if isinstance(item, Tensor) and item.dim() == 0:
                    cat_dim = None

                # Add a batch dimension to items whose `cat_dim` is `None`:
                if isinstance(item, Tensor) and cat_dim is None:
                    cat_dim = 0  # Concatenate along this new batch dimension.
                    item = item.unsqueeze(0)
                    device = item.device
                elif isinstance(item, Tensor):
                    size = item.size(cat_dim)
                    device = item.device

                batch[key].append(item)  # Append item to the attribute list.

                inc = data.__inc__(key, item)
                if isinstance(inc, (tuple, list)):
                    inc = torch.tensor(inc)
                cumsum[key].append(inc + cumsum[key][-1])

                if key in follow_batch:
                    if isinstance(size, Tensor):
                        for j, size in enumerate(size.tolist()):
                            tmp = f"{key}_{j}_batch"
                            batch[tmp] = [] if i == 0 else batch[tmp]
                            batch[tmp].append(torch.full((size,), i, dtype=torch.long, device=device))
                    else:
                        tmp = f"{key}_batch"
                        batch[tmp] = [] if i == 0 else batch[tmp]
                        batch[tmp].append(torch.full((size,), i, dtype=torch.long, device=device))

            num_nodes = data.num_nodes
            if num_nodes is not None:
                item = torch.full((num_nodes,), i, dtype=torch.long, device=device)
                batch.batch.append(item)
                batch.ptr.append(batch.ptr[-1] + num_nodes)

        batch.batch = None if len(batch.batch) == 0 else batch.batch
        batch.ptr = None if len(batch.ptr) == 1 else batch.ptr

        ref_data = data_list[0]
        for key in batch.keys:
            items = batch[key]
            item = items[0]
            cat_dim = ref_data.__cat_dim__(key, item)
            cat_dim = 0 if cat_dim is None else cat_dim
            if isinstance(item, Tensor):
                batch[key] = torch.cat(items, cat_dim)
            elif isinstance(item, (int, float)):
                batch[key] = torch.tensor(items)

        return batch.contiguous()
