"""
Trimmed-down pytorch_geometric: https://github.com/pyg-team/pytorch_geometric.
Copyright (c) 2023 PyG Team, MIT License (see NOTICE).

Local FlashCart modifications are marked with "# FlashCart:" comments.
"""

from typing import Optional, Sequence

import numpy as np
import torch.utils.data
from torch import Tensor

from .data import Data


class Dataset(torch.utils.data.Dataset):
    r"""Dataset base class for creating graph datasets.

    Subclasses implement :meth:`len` (number of examples) and :meth:`get` (data object
    at an index).
    """

    def len(self) -> int:
        """Return the underlying dataset size before applying an index selection.

        Returns:
            int: Number of examples supplied by the subclass.
        """
        raise NotImplementedError

    def get(self, idx: int) -> Data:
        """Read one graph using an index in the underlying dataset.

        Args:
            idx (int): Underlying dataset index, after resolving any index selection.

        Returns:
            Data: Graph supplied by the subclass at the requested index.
        """
        raise NotImplementedError

    def __init__(self):
        super().__init__()
        self._indices: Optional[Sequence] = None

    def indices(self) -> Sequence:
        """Return the underlying indices used to access examples in this dataset.

        Returns:
            Sequence: Stored index selection, or a range covering the full dataset.
        """
        return range(self.len()) if self._indices is None else self._indices

    def __len__(self) -> int:
        r"""The number of examples in the dataset."""
        return len(self.indices())

    def __getitem__(self, idx) -> Data:
        r"""Returns the data object at index :obj:`idx`."""
        if (
            isinstance(idx, (int, np.integer))
            or (isinstance(idx, Tensor) and idx.dim() == 0)
            or (isinstance(idx, np.ndarray) and np.isscalar(idx))
        ):
            return self.get(self.indices()[idx])
        raise IndexError(f"Only integer indexing is supported (got '{type(idx).__name__}').")

    def __repr__(self) -> str:
        arg_repr = str(len(self)) if len(self) > 1 else ""
        return f"{self.__class__.__name__}({arg_repr})"
