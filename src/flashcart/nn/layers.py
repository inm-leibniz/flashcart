from typing import Iterator, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashcart import o3
from flashcart.utils.parameter_groups import iter_child_special_parameters
from flashcart.utils.torch_geometric.dataloader import DataLoader


class RescaledSiLULayer(nn.Module):
    """SiLU rescaled to preserve the second moment of standard-normal inputs.

    Args:
        scale (float, optional): Rescaling factor. Default: None, using
            ``1 / sqrt(E[silu(x)**2])`` for standard-normal ``x``.
    """

    def __init__(self, scale: Optional[float] = None):
        super().__init__()
        if scale is None:
            scale = 1.6765324703310907
        self.register_buffer(
            "scale",
            torch.tensor(scale, dtype=torch.get_default_dtype()),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the rescaled SiLU activation elementwise.

        Args:
            x (torch.Tensor): Input values of any shape.

        Returns:
            torch.Tensor: Activated values with the input shape.
        """
        return self.scale * F.silu(x)

    def non_decayable_parameters(self) -> Iterator[nn.Parameter]:
        return iter(())

    def non_muon_parameters(self) -> Iterator[nn.Parameter]:
        return iter(())


class RescaledSigmoidLayer(nn.Module):
    """Sigmoid rescaled to preserve the second moment of standard-normal inputs.

    Args:
        scale (float, optional): Rescaling factor. Default: None, using
            ``1 / sqrt(E[sigmoid(x)**2])`` for standard-normal ``x``.
    """

    def __init__(self, scale: Optional[float] = None):
        super().__init__()
        if scale is None:
            scale = 1.8462285453386054
        self.register_buffer(
            "scale",
            torch.tensor(scale, dtype=torch.get_default_dtype()),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the rescaled sigmoid activation elementwise.

        Args:
            x (torch.Tensor): Input values of any shape.

        Returns:
            torch.Tensor: Activated values with the input shape.
        """
        return self.scale * torch.sigmoid(x)

    def non_decayable_parameters(self) -> Iterator[nn.Parameter]:
        return iter(())

    def non_muon_parameters(self) -> Iterator[nn.Parameter]:
        return iter(())


class LinearLayer(nn.Module):
    """Linear map with input-width normalization.

    Weights are initialized from a standard normal distribution and scaled by
    ``1 / sqrt(in_features)`` during evaluation. The optional bias is initialized to
    zero and excluded from weight decay.

    Args:
        in_features (int): Input width.
        out_features (int): Output width.
        bias (bool, optional): Add a bias. Default: False.
        weight_scale (float, optional): Scale applied to the weights. Default: None,
            using ``1 / sqrt(in_features)``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        weight_scale: Optional[float] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        scale = float(in_features) ** -0.5 if weight_scale is None else float(weight_scale)
        self.register_buffer(
            "scale",
            torch.tensor(scale, dtype=torch.get_default_dtype()),
            persistent=False,
        )
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the normalized linear map.

        Args:
            x (torch.Tensor): Input features with final dimension ``in_features``.

        Returns:
            torch.Tensor: Output features with final dimension ``out_features`` and
                unchanged leading dimensions.
        """
        return F.linear(x, self.weight * self.scale, self.bias)

    def non_decayable_parameters(self) -> Iterator[nn.Parameter]:
        if self.bias is not None:
            yield self.bias

    def non_muon_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.non_decayable_parameters()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, scale={float(self.scale):.6g})"
        )


def _recompute_on_unpack(
    target: torch.Tensor, last_linear: nn.Module, hidden: torch.Tensor
) -> torch.autograd.graph.saved_tensors_hooks:
    """Recompute one radial linear output when autograd restores saved tensors.

    The hook replaces references to the target tensor with a marker. Restoring that
    marker evaluates ``last_linear(hidden)`` without recording another graph.
    Hidden activations remain stored.

    Args:
        target (torch.Tensor): Linear output identified by its storage and shape.
        last_linear (nn.Module): Final radial linear map.
        hidden (torch.Tensor): Saved input to the linear map.

    Returns:
        torch.autograd.graph.saved_tensors_hooks: Context manager installing the
            save and restore hooks for the target tensor.
    """
    marker = object()
    target_ptr = target.untyped_storage().data_ptr()
    target_shape = target.shape

    def pack(t: torch.Tensor):
        if t.shape == target_shape and t.untyped_storage().data_ptr() == target_ptr:
            return marker
        return t

    def unpack(x):
        if x is marker:
            with torch.no_grad():
                return last_linear(hidden)
        return x

    return torch.autograd.graph.saved_tensors_hooks(pack, unpack)


class InteractionLayer(nn.Module):
    """Equivariant convolution with distance-dependent tensor-product weights.

    A linear map transforms the node features before coupling them with edge attributes.
    A radial multilayer perceptron provides the coupling weights, optionally conditioned
    on the sender and receiver elements. Its inputs are multiplied by the cutoff
    envelope. Messages are summed at each receiver, divided by ``sqrt(avg_neighbors)``,
    and transformed by an equivariant linear map. An optional residual projection and
    normalization complete the layer.

    Args:
        in1_l_max (int): Maximum tensor rank of the input node features.
        in2_l_max (int): Maximum tensor rank of the edge attributes.
        out_l_max (int): Maximum tensor rank of the output features.
        in1_features (int): Input feature channels per tensor rank.
        out_features (int): Output feature channels per tensor rank.
        n_radial (int): Number of radial basis functions.
        hidden_radial (list[int], optional): Radial-MLP hidden widths. Default: [64,
            64].
        n_elements (int, optional): Number of elements for element-conditioned radial
            weights. None is element-agnostic. Default: None.
        element_features (int, optional): Width of the sender and receiver element
            embeddings. Default: None, allowed only when element conditioning is
            disabled.
        avg_neighbors (float, optional): Neighbor-count normalization. Default: 1.0.
        use_sc (bool, optional): Skip connection (projected residual). Default: False.
        use_layer_norm (bool, optional): RMS layer norm on the output. Default: False.
        use_triton (bool, optional): Triton tensor-product kernels. Default: False.
        recompute_radial (bool, optional): Recompute the final radial linear output
            during differentiation instead of retaining that tensor. Hidden activations
            remain stored. Default: False.
    """

    def __init__(
        self,
        in1_l_max: int,
        in2_l_max: int,
        out_l_max: int,
        in1_features: int,
        out_features: int,
        n_radial: int,
        hidden_radial: Optional[List[int]] = None,
        n_elements: Optional[int] = None,
        element_features: Optional[int] = None,
        avg_neighbors: float = 1.0,
        use_sc: bool = False,
        use_layer_norm: bool = False,
        use_triton: bool = False,
        recompute_radial: bool = False,
    ):
        super().__init__()
        self.in1_l_max = in1_l_max
        self.in2_l_max = in2_l_max
        self.out_l_max = out_l_max
        self.in1_features = in1_features
        self.out_features = out_features
        self.n_radial = n_radial
        self.hidden_radial = [64, 64] if hidden_radial is None else list(hidden_radial)
        self.use_sc = use_sc
        self.use_layer_norm = use_layer_norm
        self.recompute_radial = recompute_radial
        if avg_neighbors <= 0.0:
            raise ValueError(f"avg_neighbors must be positive. Provided: {avg_neighbors}.")
        self.register_buffer(
            "sqrt_avg_neighbors",
            torch.tensor(float(avg_neighbors) ** 0.5, dtype=torch.get_default_dtype()),
        )

        self.out_dim = sum((2 * l + 1) * out_features for l in range(out_l_max + 1))

        out_paths, n_total_paths = o3.utils.count_output_paths(in1_l_max, in2_l_max, out_l_max)

        if n_elements is not None and element_features is None:
            raise ValueError("element_features must be provided when n_elements is set.")
        self.element_features = element_features
        radial_in = n_radial + 2 * self.element_features if n_elements is not None else n_radial
        layers = []
        for in_size, out_size in zip(
            [radial_in] + self.hidden_radial,
            self.hidden_radial + [n_total_paths * in1_features],
        ):
            layers.append(LinearLayer(in_size, out_size))
            layers.append(RescaledSiLULayer())
        self.radial_mlp = nn.Sequential(*layers[:-1])

        self.tp = o3.TensorProduct(
            in1_l_max=in1_l_max,
            in2_l_max=in2_l_max,
            out_l_max=out_l_max,
            in1_features=in1_features,
            in2_features=1,
            use_triton=use_triton,
            reduce_paths=False,
        )

        self.linear_first = o3.LinearLayer(in1_l_max, in1_l_max, in1_features, in1_features, use_triton=use_triton)
        self.linear_second = o3.LinearLayer(
            out_l_max, out_l_max, in1_features, out_features, in_paths=out_paths, use_triton=use_triton
        )
        if self.use_sc:
            self.skip_projection = (
                LinearLayer(in1_features, out_features)
                if in1_l_max == 0
                else o3.LinearLayer(
                    in1_l_max,
                    out_l_max,
                    in1_features,
                    out_features,
                    use_triton=use_triton,
                )
            )
        else:
            self.skip_projection = None
        self.layer_norm = (
            RMSLayerNorm(out_l_max, out_features, eps=1.0e-6, use_triton=use_triton) if self.use_layer_norm else None
        )

        if n_elements is not None:
            self.emb_i = nn.Embedding(n_elements, self.element_features)
            self.emb_j = nn.Embedding(n_elements, self.element_features)
            nn.init.uniform_(self.emb_i.weight, a=-0.001, b=0.001)
            nn.init.uniform_(self.emb_j.weight, a=-0.001, b=0.001)
        else:
            self.emb_i = None
            self.emb_j = None

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        envelope: torch.Tensor,
        atom_types: Optional[torch.Tensor] = None,
        out_nodes: Optional[int] = None,
    ) -> torch.Tensor:
        """Aggregate tensor-product messages onto the receiver nodes.

        Args:
            node_feats (torch.Tensor): Node features with shape
                ``(n_nodes, (in1_l_max + 1)**2 * in1_features)``.
            edge_attrs (torch.Tensor): Edge attributes with shape
                ``(n_edges, (in2_l_max + 1)**2)``.
            edge_feats (torch.Tensor): Radial features with shape
                ``(n_edges, n_radial)``.
            edge_index (torch.Tensor): Shape ``(2, n_edges)``. Row 0 contains receivers
                and row 1 contains senders.
            envelope (torch.Tensor): Cutoff values with shape ``(n_edges,)``.
            atom_types (torch.Tensor, optional): Element indices with shape
                ``(n_nodes,)``, required for element-conditioned interactions. Default:
                None.
            out_nodes (int, optional): Number of receiver nodes when the senders include
                ghost atoms. Default: None, retaining all nodes.

        Returns:
            torch.Tensor: Receiver features with shape
                ``(n_receivers, (out_l_max + 1)**2 * out_features)``, where
                ``n_receivers`` is ``out_nodes`` when supplied and otherwise
                ``n_nodes``.
        """
        idx_i, idx_j = edge_index[0], edge_index[1]
        residual_feats = node_feats if out_nodes is None else node_feats[:out_nodes]

        if self.emb_i is not None:
            if atom_types is None:
                raise ValueError(f"atom_types required when n_elements is set. Provided: {atom_types}.")
            emb_i, emb_j = self.emb_i(atom_types)[idx_i], self.emb_j(atom_types)[idx_j]
            mlp_input = torch.cat([edge_feats, emb_i, emb_j], dim=-1) * envelope.unsqueeze(-1)
        else:
            mlp_input = edge_feats * envelope.unsqueeze(-1)

        node_feats = self.linear_first(node_feats)

        if self.recompute_radial and torch.is_grad_enabled():
            hidden = self.radial_mlp[:-1](mlp_input)
            weights = self.radial_mlp[-1](hidden)
            with _recompute_on_unpack(weights, self.radial_mlp[-1], hidden):
                aggregated = self.tp(node_feats, edge_attrs, weights, idx_i, idx_j)
        else:
            weights = self.radial_mlp(mlp_input)
            aggregated = self.tp(node_feats, edge_attrs, weights, idx_i, idx_j)
        if out_nodes is not None:
            aggregated = aggregated[:out_nodes]
        aggregated = aggregated / self.sqrt_avg_neighbors

        out = self.linear_second(aggregated)
        if self.skip_projection is not None:
            skip = self.skip_projection(residual_feats)
            if self.in1_l_max == 0:
                out[:, : self.out_features] = out[:, : self.out_features] + skip
            else:
                out = out + skip
        if self.layer_norm is not None:
            out = self.layer_norm(out)

        return out

    def set_avg_neighbors(self, avg_neighbors: float) -> None:
        """Overwrite the neighbor normalization (called by ``pre_fit``).

        Args:
            avg_neighbors (float): Positive average neighbor count.
        """
        if avg_neighbors <= 0.0:
            raise ValueError(f"avg_neighbors must be positive. Provided: {avg_neighbors}.")
        self.sqrt_avg_neighbors.fill_(float(avg_neighbors) ** 0.5)

    def non_decayable_parameters(self) -> Iterator[nn.Parameter]:
        yield from iter_child_special_parameters(self, "non_decayable_parameters")

    def non_muon_parameters(self) -> Iterator[nn.Parameter]:
        yield from iter_child_special_parameters(self, "non_muon_parameters")
        if self.emb_i is not None:
            yield from self.emb_i.parameters()
            yield from self.emb_j.parameters()

    def __repr__(self) -> str:
        element_conditioned = self.emb_i is not None
        return (
            f"{self.__class__.__name__}("
            f"in1_l_max={self.in1_l_max}, in2_l_max={self.in2_l_max}, out_l_max={self.out_l_max}, "
            f"in1_features={self.in1_features}, out_features={self.out_features}, "
            f"hidden_radial={self.hidden_radial}, element_conditioned={element_conditioned}, "
            f"element_features={self.element_features}, "
            f"use_sc={self.use_sc}, use_layer_norm={self.use_layer_norm}, "
            f"out_dim={self.out_dim})"
        )


class RMSLayerNorm(nn.Module):
    """Normalize irreducible Cartesian features using an invariant mean square.

    For each tensor rank, a scalar tensor product computes the invariant squared norm.
    The result is divided by the number of components and averaged over ranks and
    feature channels. This accounts for the non-orthonormal component layout at ranks of
    two and above.

    Optional centering subtracts the channel mean from the scalar block. Learnable gains
    act separately on each rank and channel. A scalar bias is included when both affine
    transformation and centering are enabled.

    Args:
        l_max (int): Maximum tensor rank of the features.
        n_features (int): Feature channels per tensor rank.
        eps (float, optional): Stabilizer added to the mean square. Default: 1e-12.
        affine (bool, optional): Per-rank learnable gains (and an l=0 bias when
            centering). Default: True.
        centering (bool, optional): Mean-center the l=0 block. Default: False.
        use_triton (bool, optional): Triton tensor-product kernels. Default: False.
    """

    def __init__(
        self,
        l_max: int,
        n_features: int,
        eps: float = 1.0e-12,
        affine: bool = True,
        centering: bool = False,
        use_triton: bool = False,
    ):
        super().__init__()
        self.l_max = l_max
        self.n_features = n_features
        self.eps = eps
        self.affine = affine
        self.centering = centering

        self.slices = o3.utils.get_irreps_slices(l_max, n_features)
        self.shapes = o3.utils.get_irreps_shapes(l_max, n_features)
        self.block_dims = [shape[0] * shape[1] for shape in self.shapes]
        self.dim = sum(self.block_dims)

        self.tp = o3.TensorProduct(
            in1_l_max=l_max,
            in2_l_max=l_max,
            out_l_max=0,
            in1_features=n_features,
            in2_features=n_features,
            use_triton=use_triton,
            symmetric_product=True,
            shared_weights=True,
            reduce_paths=False,
        )
        _, n_paths = o3.utils.count_output_paths(l_max, l_max, 0, symmetric_product=True)
        assert n_paths == l_max + 1
        self.register_buffer("dummy_tp_weights", torch.ones(n_paths * n_features), persistent=False)
        self.register_buffer(
            "l_weights",
            torch.tensor(
                [1.0 / ((2 * l + 1) * (l_max + 1)) for l in range(l_max + 1)],
                dtype=torch.float64,
            ),
            persistent=False,
        )

        if affine:
            self.weights = nn.ParameterList([nn.Parameter(torch.ones(n_features)) for _ in range(l_max + 1)])
            self.register_buffer(
                "l_expand_index",
                torch.arange(l_max + 1).repeat_interleave(torch.tensor([2 * l + 1 for l in range(l_max + 1)])),
                persistent=False,
            )
            if centering:
                self.bias = nn.Parameter(torch.zeros(n_features))
            else:
                self.register_parameter("bias", None)
        else:
            self.weights = nn.ParameterList()
            self.register_parameter("bias", None)

    def forward(self, node_feats: torch.Tensor) -> torch.Tensor:
        """Normalize the features of each node independently.

        Args:
            node_feats (torch.Tensor): Features with shape
                ``(n_nodes, (l_max + 1)**2 * n_features)``.

        Returns:
            torch.Tensor: Normalized features with the input shape.
        """
        if not node_feats.shape[-1] == self.dim:
            raise RuntimeError(
                f"Incorrect last dimension for node_feats. " f"Expected {self.dim}, got {node_feats.shape[-1]}."
            )

        n_batch = node_feats.shape[0]

        feats = node_feats
        if self.centering:
            l0 = node_feats[:, : self.n_features]
            l0 = l0 - l0.mean(dim=-1, keepdim=True)
            feats = torch.cat([l0, node_feats[:, self.n_features :]], dim=-1)

        inv = self.tp(feats, feats, self.dummy_tp_weights).view(n_batch, self.l_max + 1, self.n_features)
        mean_sq = (inv * self.l_weights.to(inv.dtype).view(1, -1, 1)).sum(dim=1).mean(dim=-1, keepdim=True)
        scale = torch.pow(mean_sq + self.eps, -0.5)

        scaled = feats * scale
        if not self.affine:
            return scaled
        w_full = torch.stack(tuple(self.weights)).index_select(0, self.l_expand_index).view(1, -1)
        out = scaled * w_full
        if self.bias is not None:
            out = torch.cat([out[:, : self.n_features] + self.bias, out[:, self.n_features :]], dim=-1)
        return out

    def non_decayable_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.parameters()

    def non_muon_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.parameters()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(l_max={self.l_max}, n_features={self.n_features}, "
            f"eps={self.eps}, affine={self.affine}, centering={self.centering})"
        )


class EquivariantGatedLayer(nn.Module):
    """Equivariant gated residual layer initialized to the identity.

    The input scalar channels produce one gate for each tensor rank and hidden channel.
    The gated values pass through an output linear map whose weights are initialized to
    zero. The layer therefore initially returns its input. Sigmoid gates are bounded by
    their rescaling factor.

    Args:
        l_max (int): Maximum tensor rank of the features.
        in_features (int): Feature channels per tensor rank.
        hidden_features (int, optional): Width of the gated branch. Default:
            in_features.
        gate_activation (str, optional): "silu" or "sigmoid" (bounded). Default: "silu".
        gate_bias (bool, optional): Include a bias in the linear map that produces the
            gates. Default: True.
        use_triton (bool, optional): Triton kernels. Default: False.
    """

    def __init__(
        self,
        l_max: int,
        in_features: int,
        hidden_features: Optional[int] = None,
        gate_activation: str = "silu",
        gate_bias: bool = True,
        use_triton: bool = False,
    ):
        super().__init__()
        if gate_activation not in ("silu", "sigmoid"):
            raise ValueError(f"gate_activation must be 'silu' or 'sigmoid'. Provided: {gate_activation!r}.")
        self.l_max = l_max
        self.in_features = in_features
        self.gate_activation_name = gate_activation
        self.gate_bias = gate_bias
        self.hidden_features = in_features if hidden_features is None else hidden_features
        if self.hidden_features <= 0:
            raise ValueError(f"hidden_features must be positive. Provided: {self.hidden_features}.")

        self.in_dim = sum((2 * l + 1) * in_features for l in range(l_max + 1))
        self.hidden_dim = sum((2 * l + 1) * self.hidden_features for l in range(l_max + 1))
        self.hidden_slices = o3.utils.get_irreps_slices(l_max, self.hidden_features)

        self.value_linear = o3.LinearLayer(
            l_max,
            l_max,
            in_features,
            self.hidden_features,
            use_triton=use_triton,
        )
        self.gate_linear = LinearLayer(
            in_features,
            (l_max + 1) * self.hidden_features,
            bias=gate_bias,
        )
        self.gate_activation = RescaledSiLULayer() if gate_activation == "silu" else RescaledSigmoidLayer()
        self.output_linear = o3.LinearLayer(
            l_max,
            l_max,
            self.hidden_features,
            in_features,
            use_triton=use_triton,
        )
        for weight in self.output_linear.weight:
            nn.init.zeros_(weight)

    def forward(self, node_feats: torch.Tensor) -> torch.Tensor:
        """Add the gated equivariant update to the input features.

        Args:
            node_feats (torch.Tensor): Features with shape
                ``(n_nodes, (l_max + 1)**2 * in_features)``.

        Returns:
            torch.Tensor: Updated features with the input shape.
        """
        if node_feats.shape[-1] != self.in_dim:
            raise RuntimeError(
                f"Incorrect last dimension for node_feats. Expected {self.in_dim}, got {node_feats.shape[-1]}."
            )

        values = self.value_linear(node_feats)
        scalar_feats = node_feats[:, : self.in_features]
        gates = self.gate_activation(self.gate_linear(scalar_feats))
        gates = gates.view(node_feats.shape[0], self.l_max + 1, self.hidden_features)

        gated_values = []
        for l, (start, stop) in enumerate(self.hidden_slices):
            block = values[:, start:stop].view(node_feats.shape[0], 2 * l + 1, self.hidden_features)
            gated_values.append((block * gates[:, l].unsqueeze(1)).flatten(1))

        update = self.output_linear(torch.cat(gated_values, dim=-1))
        return node_feats + update

    def non_decayable_parameters(self) -> Iterator[nn.Parameter]:
        yield from iter_child_special_parameters(self, "non_decayable_parameters")

    def non_muon_parameters(self) -> Iterator[nn.Parameter]:
        yield from iter_child_special_parameters(self, "non_muon_parameters")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"l_max={self.l_max}, in_features={self.in_features}, "
            f"hidden_features={self.hidden_features})"
        )


class IrrepsSelector(nn.Module):
    """Select tensor-rank blocks up to ``out_l_max``.

    Args:
        in_l_max (int): Maximum tensor rank of the input layout.
        out_l_max (int): Maximum tensor rank kept (<= in_l_max).
        in_features (int): Feature channels per tensor rank.
        in_paths (list[int], optional): Paths per tensor rank of the input layout.
            Default: one path per tensor rank.
    """

    def __init__(
        self,
        in_l_max: int,
        out_l_max: int,
        in_features: int,
        in_paths: Optional[List[int]] = None,
    ):
        super().__init__()
        if out_l_max > in_l_max:
            raise ValueError(
                "out_l_max must be less than or equal to in_l_max. "
                f"Provided: out_l_max={out_l_max}, in_l_max={in_l_max}."
            )

        in_paths = [1] * (in_l_max + 1) if in_paths is None else list(in_paths)
        slices = o3.utils.get_irreps_slices(in_l_max, in_features, in_paths)
        self.stop = slices[out_l_max][1]
        self.in_dim = sum((2 * l + 1) * in_features * in_paths[l] for l in range(in_l_max + 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Select the leading tensor-rank blocks.

        Args:
            x (torch.Tensor): Features with shape ``(n_nodes, in_dim)`` in increasing
                rank order.

        Returns:
            torch.Tensor: A view containing ranks through ``out_l_max``.
        """
        if x.shape[-1] != self.in_dim:
            raise RuntimeError(f"Incorrect last dimension for x. Expected {self.in_dim}, got {x.shape[-1]}.")
        return x[:, : self.stop]

    def non_decayable_parameters(self) -> Iterator[nn.Parameter]:
        return iter(())

    def non_muon_parameters(self) -> Iterator[nn.Parameter]:
        return iter(())


class ProductPathWeights(nn.Module):
    """Multiplicative per-path weights ``1 + delta`` for product tensor products.

    ``delta`` is zero-initialized, so the weights start at exactly one. It is a fixed
    buffer when not learnable, a shared parameter, or (when ``n_elements`` is set) a
    per-element embedding queried by atom type (element-dependent weights are
    necessarily learnable).

    Args:
        n_weights (int): Number of weights (paths times features).
        n_elements (int, optional): Number of elements for element-dependent weights.
            Requires ``learnable``. Default: None.
        learnable (bool, optional): Train the deltas. Default: False.
    """

    def __init__(
        self,
        n_weights: int,
        n_elements: Optional[int] = None,
        learnable: bool = False,
    ):
        super().__init__()
        if n_elements is not None and not learnable:
            raise ValueError(
                "Element-dependent path weights must be learnable. Provided: learnable=False with n_elements set."
            )
        self.learnable = learnable
        self.element_dependent = False
        if not self.learnable:
            self.register_buffer("delta", torch.zeros(n_weights), persistent=False)
        elif n_elements is None:
            self.delta = nn.Parameter(torch.zeros(n_weights))
        else:
            self.element_dependent = True
            self.delta = nn.Embedding(n_elements, n_weights)
            nn.init.zeros_(self.delta.weight)

    def forward(self, atom_types: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return shared or element-dependent coupling weights.

        Args:
            atom_types (torch.Tensor, optional): Element indices with shape
                ``(n_nodes,)``, required for element-dependent weights. Default: None.

        Returns:
            torch.Tensor: Weights with shape ``(n_weights,)`` when shared or
                ``(n_nodes, n_weights)`` when element-dependent.
        """
        if self.element_dependent:
            if atom_types is None:
                raise ValueError("atom_types required when n_elements is set.")
            return 1.0 + self.delta(atom_types)
        return 1.0 + self.delta

    def non_decayable_parameters(self) -> Iterator[nn.Parameter]:
        return iter(())

    def non_muon_parameters(self) -> Iterator[nn.Parameter]:
        if not self.learnable:
            return
        if self.element_dependent:
            yield from self.delta.parameters()
        else:
            yield self.delta


class ProductLayer(nn.Module):
    """Construct repeated equivariant products of the aggregated node features.

    The recurrence starts with a linear contribution and forms terms of polynomial
    degree up to ``correlation`` by multiplying the current features with the original
    input. Intermediate states retain ranks up to ``in_l_max``. Each contribution to the
    output retains ranks up to ``out_l_max``.

    With ``path_reduced=True``, each tensor product sums its coupling paths and
    normalizes each output rank by the square root of its path count. The
    contributions from all degrees are added before a final linear map. Otherwise,
    each recurrence step uses a linear map to mix its retained paths and channels.
    Path weights are learnable when paths are reduced or depend on the element.
    An optional residual projection and normalization complete the layer.

    Args:
        in_l_max (int): Maximum tensor rank of the input features.
        out_l_max (int): Maximum tensor rank of the output features.
        in_features (int): Feature channels per tensor rank.
        n_elements (int, optional): Number of elements for element-dependent path
            weights. None is element-agnostic. Default: None.
        correlation (int, optional): Maximum polynomial degree of the recurrence in the
            input features. Must be at least 2. Default: 2.
        path_reduced (bool, optional): Sum coupling paths inside the kernel. Default:
            False.
        use_sc (bool, optional): Skip connection. Default: False.
        use_layer_norm (bool, optional): RMS layer norm on the output. Default: False.
        use_triton (bool, optional): Triton kernels. Default: False.
    """

    def __init__(
        self,
        in_l_max: int,
        out_l_max: int,
        in_features: int,
        n_elements: Optional[int] = None,
        correlation: int = 2,
        path_reduced: bool = False,
        use_sc: bool = False,
        use_layer_norm: bool = False,
        use_triton: bool = False,
    ):
        super().__init__()
        if correlation < 2:
            raise ValueError(
                "ProductLayer requires correlation >= 2. "
                "Use FlashCartPotential(correlation=1) for an interaction-only model."
            )
        self.in_l_max = in_l_max
        self.out_l_max = out_l_max
        self.in_features = in_features
        self.correlation = correlation
        self.path_reduced = path_reduced
        self.learnable_path_weights = path_reduced or n_elements is not None
        self.use_sc = use_sc
        self.use_layer_norm = use_layer_norm

        self.in_dim = sum((2 * l + 1) * in_features for l in range(in_l_max + 1))
        self.out_dim = sum((2 * l + 1) * in_features for l in range(out_l_max + 1))
        self.skip_projection = (
            o3.LinearLayer(
                in_l_max,
                out_l_max,
                in_features,
                in_features,
                use_triton=use_triton,
            )
            if self.use_sc
            else None
        )
        self.layer_norm = (
            RMSLayerNorm(out_l_max, in_features, eps=1.0e-6, use_triton=use_triton) if self.use_layer_norm else None
        )

        if self.path_reduced:
            self.final_linear = o3.LinearLayer(
                out_l_max,
                out_l_max,
                in_features,
                in_features,
                use_triton=use_triton,
            )

        self.tensor_products = nn.ModuleList()
        self.path_weights = nn.ModuleList()
        self.irreps_selectors = nn.ModuleList()
        self.product_linears = nn.ModuleList()
        self.path_norm_names: List[Optional[str]] = []

        for product_idx in range(self.correlation):
            product_out_l_max = out_l_max if product_idx == self.correlation - 1 else in_l_max
            if product_idx == 0:
                in1_l_max = in_l_max
                in2_l_max = 0
                in2_features = 1
            else:
                in1_l_max = in_l_max
                in2_l_max = in_l_max
                in2_features = in_features

            out_paths, n_total_paths = o3.utils.count_output_paths(
                in1_l_max, in2_l_max, product_out_l_max, symmetric_product=False
            )

            self.path_weights.append(
                ProductPathWeights(
                    n_total_paths * in_features,
                    n_elements=n_elements,
                    learnable=self.learnable_path_weights,
                )
            )

            if path_reduced:
                path_norm = torch.cat(
                    [
                        torch.full(
                            ((2 * l + 1) * in_features,),
                            out_paths[l] ** -0.5,
                        )
                        for l in range(product_out_l_max + 1)
                    ]
                )
                if torch.all(path_norm == 1.0):
                    self.path_norm_names.append(None)
                else:
                    path_norm_name = f"path_norm_{product_idx}"
                    self.register_buffer(path_norm_name, path_norm)
                    self.path_norm_names.append(path_norm_name)
            else:
                self.product_linears.append(
                    o3.LinearLayer(
                        product_out_l_max,
                        product_out_l_max if product_idx < self.correlation - 1 else out_l_max,
                        in_features,
                        in_features,
                        in_paths=out_paths,
                        use_triton=use_triton,
                    )
                )

            self.irreps_selectors.append(
                (
                    nn.Identity()
                    if product_out_l_max == out_l_max
                    else IrrepsSelector(
                        product_out_l_max,
                        out_l_max,
                        in_features,
                    )
                )
            )

            if product_idx > 0:
                self.tensor_products.append(
                    o3.TensorProduct(
                        in1_l_max=in1_l_max,
                        in2_l_max=in2_l_max,
                        out_l_max=product_out_l_max,
                        in1_features=in_features,
                        in2_features=in2_features,
                        use_triton=use_triton,
                        reduce_paths=path_reduced,
                        symmetric_product=False,
                        shared_weights=n_elements is None,
                    )
                )

        if self.learnable_path_weights:
            self.register_buffer(
                "step0_weight_index",
                torch.cat(
                    [
                        torch.arange(l * in_features, (l + 1) * in_features).repeat(2 * l + 1)
                        for l in range(in_l_max + 1)
                    ]
                ),
                persistent=False,
            )
        else:
            self.step0_weight_index = None

    def forward(
        self,
        node_feats: torch.Tensor,
        atom_types: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sum the contributions from the repeated equivariant products.

        Args:
            node_feats (torch.Tensor): Node features with shape
                ``(n_nodes, (in_l_max + 1)**2 * in_features)``.
            atom_types (torch.Tensor, optional): Element indices with shape
                ``(n_nodes,)``, required for element-dependent path weights.
                Default: None.

        Returns:
            torch.Tensor: Output features with shape
                ``(n_nodes, (out_l_max + 1)**2 * in_features)``.
        """
        output_feats = node_feats.new_zeros(node_feats.shape[0], self.out_dim)

        product_feats = node_feats
        for step_idx in range(self.correlation):
            if step_idx == 0:
                if self.step0_weight_index is not None:
                    weights = self.path_weights[0](atom_types)
                    product_feats = node_feats * weights[..., self.step0_weight_index]
                else:
                    product_feats = node_feats
            else:
                weights = self.path_weights[step_idx](atom_types)
                product_feats = self.tensor_products[step_idx - 1](product_feats, node_feats, weights)
            if self.path_reduced:
                path_norm_name = self.path_norm_names[step_idx]
                if path_norm_name is not None:
                    product_feats = product_feats * getattr(self, path_norm_name)
            else:
                product_feats = self.product_linears[step_idx](product_feats)
            output_feats = output_feats + self.irreps_selectors[step_idx](product_feats)

        if self.path_reduced:
            output_feats = self.final_linear(output_feats)
        if self.skip_projection is not None:
            output_feats = output_feats + self.skip_projection(node_feats)
        if self.layer_norm is not None:
            output_feats = self.layer_norm(output_feats)

        return output_feats

    def non_decayable_parameters(self) -> Iterator[nn.Parameter]:
        yield from iter_child_special_parameters(self, "non_decayable_parameters")

    def non_muon_parameters(self) -> Iterator[nn.Parameter]:
        yield from iter_child_special_parameters(self, "non_muon_parameters")

    def __repr__(self) -> str:
        if self.path_weights[0].element_dependent:
            path_weights = "element"
        elif self.learnable_path_weights:
            path_weights = "learnable"
        else:
            path_weights = "fixed"
        return (
            f"{self.__class__.__name__}("
            f"in_l_max={self.in_l_max}, out_l_max={self.out_l_max}, "
            f"in_features={self.in_features}, correlation={self.correlation}, "
            f"path_reduced={self.path_reduced}, "
            f"path_weights={path_weights!r}, "
            f"use_sc={self.use_sc}, use_layer_norm={self.use_layer_norm}, "
            f"in_dim={self.in_dim}, out_dim={self.out_dim})"
        )


class EnergyReadoutLayer(nn.Module):
    """MLP readout of per-atom energies from every layer's scalar channels.

    Args:
        in_features (int): Scalar channels contributed by each layer.
        n_layers (int): Number of contributing layers.
        hidden_readout (list[int], optional): MLP hidden widths. Default: [64, 64].
    """

    def __init__(self, in_features: int, n_layers: int, hidden_readout: Optional[List[int]] = None):
        super().__init__()
        self.in_features = in_features
        self.n_layers = n_layers
        self.hidden_readout = [64, 64] if hidden_readout is None else list(hidden_readout)

        layers = []
        for in_size, out_size in zip(
            [in_features * n_layers] + self.hidden_readout,
            self.hidden_readout + [1],
        ):
            layers.append(LinearLayer(in_size, out_size))
            layers.append(RescaledSiLULayer())
        self.readout_mlp = nn.Sequential(*layers[:-1])

    def forward(self, node_feats: List[torch.Tensor]) -> torch.Tensor:
        """Predict site energies from the scalar features of all layers.

        Args:
            node_feats (list[torch.Tensor]): One feature tensor per layer, each with
                ``n_nodes`` rows and at least ``in_features`` leading scalar channels.

        Returns:
            torch.Tensor: Site energies with shape ``(n_nodes,)``.
        """
        scalars = torch.cat([nf[:, : self.in_features] for nf in node_feats], dim=1)
        if scalars.shape[-1] != self.in_features * self.n_layers:
            raise RuntimeError(
                "Incorrect last dimension for scalars. "
                f"Expected: {self.in_features * self.n_layers}. "
                f"Provided: {scalars.shape[-1]}."
            )
        node_energies = self.readout_mlp(scalars).squeeze(-1)
        return node_energies

    def non_decayable_parameters(self) -> Iterator[nn.Parameter]:
        yield from iter_child_special_parameters(self, "non_decayable_parameters")

    def non_muon_parameters(self) -> Iterator[nn.Parameter]:
        yield from iter_child_special_parameters(self, "non_muon_parameters")
        yield self.readout_mlp[-1].weight

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"in_features={self.in_features}, n_layers={self.n_layers}, "
            f"hidden_readout={self.hidden_readout})"
        )


class ScaleShiftEnergyLayer(nn.Module):
    """Scale site energies and store the atomic energy shifts.

    ``forward`` multiplies each site energy by its element-specific scale. The scales
    and shifts are stored as non-trainable buffers. Training loaders subtract the shifts
    from reference energies. Calculators may restore them when reporting energies.

    Args:
        n_elements (int): Number of elements.
    """

    def __init__(self, n_elements: int):
        super().__init__()
        self.n_elements = n_elements
        self.register_buffer("shifts", torch.zeros(n_elements))
        self.register_buffer("scales", torch.ones(n_elements))

    def forward(self, node_output: torch.Tensor, atom_types: torch.Tensor) -> torch.Tensor:
        """Apply the stored scale for each atom's element.

        Args:
            node_output (torch.Tensor): Site energies with shape ``(n_nodes,)``.
            atom_types (torch.Tensor): Element indices with shape ``(n_nodes,)``.

        Returns:
            torch.Tensor: Scaled site energies with shape ``(n_nodes,)``. Atomic shifts
                are not added.
        """
        return self.scales[atom_types] * node_output

    def initialize(self, train_loader: DataLoader, atomic_shifts=None, fit_atomic_shifts: bool = True) -> None:
        """Initialize atomic energy shifts and energy scales from the training set.

        The reference force-component RMS supplies a common scale for all elements.

        Args:
            train_loader (DataLoader): Loader over the training set.
            atomic_shifts (Sequence[float], optional): Initial per-element shifts.
                Default: None, using a zero prior when fitting. Required when
                ``fit_atomic_shifts=False``.
            fit_atomic_shifts (bool, optional): Fit residual shifts on top of the prior
                by ridge-regularized least squares on the training energies. Default:
                True.
        """
        from flashcart.data.statistics import get_atomic_energy_shifts, get_force_rms

        shifts = get_atomic_energy_shifts(
            train_loader,
            self.n_elements,
            atomic_shifts=atomic_shifts,
            fit=fit_atomic_shifts,
        )
        scales = get_force_rms(train_loader, self.n_elements)
        dtype = self.shifts.dtype
        self.shifts.copy_(torch.as_tensor(shifts, dtype=dtype))
        self.scales.copy_(torch.as_tensor(scales, dtype=dtype))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_elements={self.n_elements})"
