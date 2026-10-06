"""Couple Cartesian tensor features with weighted equivariant products."""

from typing import Optional

import torch
import torch.nn as nn

from flashcart.o3._tensor_product import py_tensor_product, triton_tensor_product
from flashcart.o3.utils import count_output_paths, get_irreps_slices


class TensorProduct(nn.Module):
    """Evaluate weighted products of irreducible Cartesian tensor features.

    Each coupling ``(l1, l2) -> l_out`` satisfying the triangle rule and an even value
    of ``l1 + l2 + l_out`` defines one path. These couplings preserve the natural parity
    ``(-1) ** l`` of rank-l tensors. Each path has a separate weight for every output
    feature channel.

    With ``reduce_paths=False``, the output retains the individual paths. Otherwise,
    their contributions are summed at each output tensor rank. Providing both ``idx_i``
    and ``idx_j`` applies the product as an edge convolution: first-input features are
    gathered from sender nodes, combined with second-input edge features, and summed at
    receiver nodes.

    Args:
        in1_l_max (int): Maximum tensor rank of the first input.
        in2_l_max (int): Maximum tensor rank of the second input.
        out_l_max (int): Maximum tensor rank of the output.
        in1_features (int): Feature channels of the first input.
        in2_features (int): Feature channels of the second input. Must equal
            in1_features unless one of the two is 1 (broadcast).
        symmetric_product (bool, optional): Require identical inputs and retain only
            couplings with ``l1 >= l2``. Default: False.
        use_triton (bool, optional): Use Triton kernels for CUDA inputs when available.
            All requested tensor ranks must then fit within the configured kernel limit.
            Default: False.
        shared_weights (bool, optional): Use one weight vector for every row or edge. If
            False, supply a separate vector for each row or edge. Default: False.
        reduce_paths (bool, optional): Sum path outputs per tensor rank instead of
            stacking them. Default: False.
    """

    def __init__(
        self,
        in1_l_max: int,
        in2_l_max: int,
        out_l_max: int,
        in1_features: int,
        in2_features: int,
        symmetric_product: bool = False,
        use_triton: bool = False,
        shared_weights: bool = False,
        reduce_paths: bool = False,
    ):
        super().__init__()

        if not (in1_features == in2_features or (in1_features == 1 or in2_features == 1)):
            raise ValueError(
                f"Input dimensions must match unless one of them is 1. "
                f"Provided: {in1_features=} and {in2_features=}."
            )

        self.in1_l_max = in1_l_max
        self.in2_l_max = in2_l_max
        self.out_l_max = out_l_max
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.symmetric_product = symmetric_product
        self.use_triton = use_triton
        self.shared_weights = shared_weights
        self.reduce_paths = reduce_paths

        self.in1_dim = sum([(2 * l + 1) * in1_features for l in range(in1_l_max + 1)])
        self.in2_dim = sum([(2 * l + 1) * in2_features for l in range(in2_l_max + 1)])

        self.in1_slices = get_irreps_slices(in1_l_max, in1_features)
        self.in2_slices = get_irreps_slices(in2_l_max, in2_features)

        self.out_paths, self.n_total_paths = count_output_paths(
            in1_l_max, in2_l_max, out_l_max, symmetric_product=symmetric_product
        )

    def forward(
        self,
        in1: torch.Tensor,
        in2: torch.Tensor,
        weights: torch.Tensor,
        idx_i: Optional[torch.Tensor] = None,
        idx_j: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Evaluate weighted tensor products, optionally summing over edges.

        Let ``features = max(in1_features, in2_features)``. Input channels are paired
        elementwise, with broadcasting when one input has a single channel. Weights
        contain one value per path and feature. The input widths are
        ``in1_dim = (in1_l_max + 1)**2 * in1_features`` and
        ``in2_dim = (in2_l_max + 1)**2 * in2_features``. Path weights follow the
        ordering described by :func:`flashcart.o3.utils.count_output_paths`.

        Args:
            in1 (torch.Tensor): First input of shape ``(n, in1_dim)``. In convolution
                mode, the rows correspond to nodes.
            in2 (torch.Tensor): Second input of shape ``(n, in2_dim)``. In convolution
                mode, its shape is ``(n_edges, in2_dim)``.
            weights (torch.Tensor): Weights of shape ``(n_total_paths * features,)``
                when ``shared_weights=True``. Otherwise, the shape is
                ``(n, n_total_paths * features)``, or
                ``(n_edges, n_total_paths * features)`` in convolution mode.
            idx_i (torch.Tensor, optional): Integer receiver-node indices of shape
                ``(n_edges,)``. Provide together with ``idx_j`` to enable convolution
                mode. Default: None.
            idx_j (torch.Tensor, optional): Integer sender-node indices of shape
                ``(n_edges,)``. Default: None.

        Returns:
            torch.Tensor: Features with one row per first-input row, including in
                convolution mode. Blocks are concatenated in increasing output tensor
                rank. Without path reduction, the rank-l block has shape
                ``(2 * l + 1, out_paths[l], features)`` before flattening. With path
                reduction, it has shape ``(2 * l + 1, features)``. Ranks with no allowed
                paths contribute no output components.
        """
        if not in1.shape[-1] == self.in1_dim:
            raise RuntimeError(f"Incorrect last dimension for in1. Expected {self.in1_dim}, got {in1.shape[-1]}.")

        if not in2.shape[-1] == self.in2_dim:
            raise RuntimeError(f"Incorrect last dimension for in2. Expected {self.in2_dim}, got {in2.shape[-1]}.")

        if self.symmetric_product and in1 is not in2:
            torch._assert_async(
                (in1 == in2).all(),
                "Symmetric product requires in1 == in2.",
            )

        if self.use_triton and in1.is_cuda:
            return triton_tensor_product(
                in1,
                in2,
                weights,
                idx_i,
                idx_j,
                self.in1_l_max,
                self.in2_l_max,
                self.out_l_max,
                self.in1_features,
                self.in2_features,
                self.in1_slices,
                self.in2_slices,
                self.symmetric_product,
                self.shared_weights,
                tuple(self.out_paths),
                self.reduce_paths,
            )
        return py_tensor_product(
            in1,
            in2,
            weights,
            idx_i,
            idx_j,
            self.in1_l_max,
            self.in2_l_max,
            self.out_l_max,
            self.in1_features,
            self.in2_features,
            self.in1_slices,
            self.in2_slices,
            self.symmetric_product,
            self.shared_weights,
            tuple(self.out_paths),
            self.reduce_paths,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.in1_l_max} x {self.in2_l_max} -> {self.out_l_max}, "
            f"in1_features={self.in1_features}, in2_features={self.in2_features}, "
            f"symmetric={self.symmetric_product}, use_triton={self.use_triton}, "
            f"paths={self.out_paths}, total_paths={self.n_total_paths}, "
            f"reduce_paths={self.reduce_paths})"
        )
