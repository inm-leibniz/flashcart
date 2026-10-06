"""Mix Cartesian tensor channels and coupling paths with equivariant linear maps."""

from typing import Optional, Sequence

import torch
import torch.nn as nn

from flashcart.o3._linear import make_linear_metadata, py_linear, triton_linear


class LinearLayer(nn.Module):
    """Mix feature channels with an equivariant linear map.

    A separate weight matrix at each tensor rank mixes the input channels and coupling
    paths. The same matrix acts on every tensor component at that rank. Weights are
    initialized from a standard normal distribution and multiplied by a fixed scale
    during evaluation.

    The map retains ranks up to ``min(in_l_max, out_l_max)``. Higher input ranks are
    discarded, and higher output ranks are filled with zeros.

    Args:
        in_l_max (int): Maximum tensor rank of the input.
        out_l_max (int): Maximum tensor rank of the output.
        in_features (int): Input feature channels per tensor rank.
        out_features (int): Output feature channels per tensor rank.
        in_paths (Sequence[int], optional): Paths per input tensor rank for
            path-resolved inputs. Default: one per tensor rank.
        weight_scale (float, optional): Scale applied to every weight matrix. Default:
            None, which uses ``1 / sqrt(in_features * in_paths[l])`` at rank ``l``.
        use_triton (bool, optional): Use Triton kernels for CUDA inputs when available
            and when the number of active tensor ranks fits within the configured kernel
            limit. Otherwise, use PyTorch. Default: False.
    """

    def __init__(
        self,
        in_l_max: int,
        out_l_max: int,
        in_features: int,
        out_features: int,
        in_paths: Optional[Sequence[int]] = None,
        weight_scale: Optional[float] = None,
        use_triton: bool = False,
    ):
        super().__init__()
        self.in_l_max = in_l_max
        self.out_l_max = out_l_max
        self.in_features = in_features
        self.out_features = out_features
        self.in_paths = [1] * (in_l_max + 1) if in_paths is None else [int(p) for p in in_paths]
        self.weight_scale = weight_scale
        self.use_triton = use_triton

        if len(self.in_paths) != in_l_max + 1:
            raise ValueError(f"Expected {in_l_max + 1} in_paths, got {len(self.in_paths)}.")

        self.common_l_max = min(in_l_max, out_l_max)
        self.in_dim = sum((2 * l + 1) * in_features * self.in_paths[l] for l in range(in_l_max + 1))
        self.out_dim = sum((2 * l + 1) * out_features for l in range(out_l_max + 1))

        meta, scale, n_weight, out_size = make_linear_metadata(
            device=torch.device("cpu"),
            in_l_max=in_l_max,
            out_l_max=out_l_max,
            in_features=in_features,
            out_features=out_features,
            in_paths=self.in_paths,
            weight_scale=weight_scale,
        )
        meta_rows = [[int(v) for v in row] for row in meta.tolist()]
        self.meta_lists = tuple([row[i] for row in meta_rows] for i in range(6))
        self.register_buffer("meta", meta, persistent=False)
        self.register_buffer("scale", scale, persistent=False)
        self.n_weight = int(n_weight)
        if int(out_size) != self.out_dim:
            raise RuntimeError(f"Metadata out_size mismatch: {out_size} != {self.out_dim}.")

        self.weight = nn.ParameterList()
        for l in range(self.common_l_max + 1):
            f_in_l = in_features * self.in_paths[l]
            f_out_l = out_features
            self.weight.append(nn.Parameter(torch.randn(f_out_l, f_in_l)))

    def _weight_views(self) -> list[torch.Tensor]:
        return list(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix input channels and paths independently at each tensor rank.

        The input width is ``sum((2*l + 1) * in_features * in_paths[l])`` over
        ranks zero to ``in_l_max``. The output width is
        ``(out_l_max + 1)**2 * out_features``.

        Args:
            x (torch.Tensor): Input features of shape ``(batch, in_dim)``, with blocks
                concatenated in increasing tensor rank.

        Returns:
            torch.Tensor: Output features of shape ``(batch, out_dim)``, with one path
                and ``out_features`` channels per tensor rank.
        """
        if x.shape[-1] != self.in_dim:
            raise RuntimeError(f"Incorrect last dimension for x. Expected {self.in_dim}, got {x.shape[-1]}.")
        weights = self._weight_views()
        if self.use_triton and x.is_cuda:
            return triton_linear(
                x,
                weights,
                self.in_l_max,
                self.out_l_max,
                self.in_features,
                self.out_features,
                in_paths=self.in_paths,
                scale=self.scale,
                precomputed_meta=self.meta,
                precomputed_meta_lists=self.meta_lists,
            )
        return py_linear(
            x,
            weights,
            self.in_l_max,
            self.out_l_max,
            self.in_features,
            self.out_features,
            in_paths=self.in_paths,
            scale=self.scale,
        )

    def extra_repr(self) -> str:
        return (
            f"in_l_max={self.in_l_max}, out_l_max={self.out_l_max}, "
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"in_paths={self.in_paths}, "
            f"out_dim={self.out_dim}, n_weights={self.n_weight}, "
            f"use_triton={self.use_triton}"
        )
