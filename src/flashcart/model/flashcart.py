from collections.abc import Mapping
from typing import Any, Dict, Iterator, List, Literal, Optional

import torch
import torch.nn as nn
from lightning.fabric import Fabric

from flashcart.model.atomistic import AtomisticModel
from flashcart.nn.layers import (
    EnergyReadoutLayer,
    EquivariantGatedLayer,
    InteractionLayer,
    ProductLayer,
    ScaleShiftEnergyLayer,
)
from flashcart.nn.radial import ChebyshevBasis, PolynomialCutoff
from flashcart import o3
from flashcart.utils.geometry import get_edge_vectors, uses_periodic_shifts
from flashcart.utils.lammps import lammps_exchange_features
from flashcart.utils.torch_geometric.dataloader import DataLoader

MIN_EDGE_LENGTH = 1.0e-4


class FlashCartPotential(AtomisticModel):
    """Equivariant potential that predicts site energies from atomic graphs.

    Each interaction layer combines neighboring node features with the edge directions
    through a tensor-product convolution. Optional gated nonlinearities and repeated
    node-local products transform the aggregated features. The scalar channels from
    every layer contribute to the energy readout, followed by per-element scaling.

    The model predicts energies relative to the stored atomic shifts. Training loaders
    subtract these shifts from the reference energies and calculators can restore them
    when reporting energies.

    Args:
        elements (list[str]): Chemical symbols in the order used by the integer
            atom-type indices.
        r_max (float): Neighbor cutoff radius.
        r_min (float, optional): Lower bound of the radial mapping interval. Default:
            0.0.
        n_hidden_feats (int, optional): Channels per tensor rank in the hidden node
            features. Default: 32.
        l_max_hidden_feats (int, optional): Maximum tensor rank of the hidden node
            features carried between layers. Default: 1.
        l_max_edge_attrs (int, optional): Maximum tensor rank of the irreducible
            Cartesian edge attributes (also the interaction output tensor rank when
            product layers are used). Default: 3.
        n_radial (int, optional): Number of Chebyshev radial basis functions. Default:
            8.
        n_interactions (int, optional): Number of message-passing layers. Default: 2.
        correlation (int, optional): Maximum polynomial degree of the repeated products
            in the aggregated node features. A value of 1 disables product layers.
            Default: 3.
        path_reduced_product (bool, optional): Path-reduced product coupling with
            learnable per-path weights (see ProductLayer). Default: True.
        element_agnostic_interaction (bool, optional): If False, interaction layers
            condition on atom types. Default: True.
        element_agnostic_product (bool, optional): If False, product layers use
            element-dependent weights. Default: True.
        n_element_interaction_features (int, optional): Element-embedding width for
            element-conditioned interactions. Default: 16.
        nonlinearity (bool, optional): Gated nonlinearity after each interaction.
            Default: False.
        gate_activation (str, optional): Activation name for the gates. Default: "silu".
        gate_bias (bool, optional): Bias in the gate-producing linear. Default: True.
        layer_norm (bool, optional): RMS layer norm inside interaction and product
            layers. Default: False.
        hidden_radial (list[int], optional): Radial-MLP hidden widths. Default: [64,
            64].
        hidden_readout (list[int], optional): Readout-MLP hidden widths. Default: [64,
            64].
        radial_mapping (str, optional): Distance mapping of the Chebyshev basis,
            "cosine" or "affine". Default: "affine".
        cutoff_poly_order (int, optional): Order of the polynomial cutoff envelope.
            Default: 6.
        avg_neighbors (float, optional): Average neighbor count used for normalization.
            Default: None, estimated during ``pre_fit``.
        atomic_shifts (list[float], optional): Initial per-element energy shifts, in
            ``elements`` order. Default: None, using a zero prior when fitting shifts.
            Required when ``fit_atomic_shifts=False``.
        fit_atomic_shifts (bool, optional): Fit the shifts from the training set during
            ``pre_fit``. Default: True.
        recompute_radial (bool, optional): Recompute the final radial linear output
            during differentiation instead of retaining that tensor. Hidden activations
            remain stored. This option is omitted from saved model configurations.
            Default: False.
        use_triton (bool, optional): Use the Triton kernels. Default: False.
    """

    def __init__(
        self,
        elements: List[str],
        r_max: float,
        r_min: float = 0.0,
        n_hidden_feats: int = 32,
        l_max_hidden_feats: int = 1,
        l_max_edge_attrs: int = 3,
        n_radial: int = 8,
        n_interactions: int = 2,
        correlation: int = 3,
        path_reduced_product: bool = True,
        element_agnostic_interaction: bool = True,
        element_agnostic_product: bool = True,
        n_element_interaction_features: int = 16,
        nonlinearity: bool = False,
        gate_activation: str = "silu",
        gate_bias: bool = True,
        layer_norm: bool = False,
        hidden_radial: Optional[List[int]] = None,
        hidden_readout: Optional[List[int]] = None,
        radial_mapping: Literal["cosine", "affine"] = "affine",
        cutoff_poly_order: int = 6,
        avg_neighbors: Optional[float] = None,
        atomic_shifts: Optional[List[float]] = None,
        fit_atomic_shifts: bool = True,
        recompute_radial: bool = False,
        use_triton: bool = False,
    ):
        super().__init__()

        self.elements = list(elements)
        self.r_min = r_min
        self.r_max = float(r_max)
        self.n_hidden_feats = n_hidden_feats
        self.l_max_hidden_feats = l_max_hidden_feats
        self.l_max_edge_attrs = l_max_edge_attrs
        self.n_radial = n_radial
        self.n_interactions = n_interactions
        self.correlation = correlation
        self.path_reduced_product = path_reduced_product
        self.hidden_radial = [64, 64] if hidden_radial is None else list(hidden_radial)
        self.hidden_readout = [64, 64] if hidden_readout is None else list(hidden_readout)
        self.radial_mapping = radial_mapping
        self.cutoff_poly_order = cutoff_poly_order
        self.atomic_shifts = None if atomic_shifts is None else list(atomic_shifts)
        self.fit_atomic_shifts = bool(fit_atomic_shifts)
        self.avg_neighbors = 1.0 if avg_neighbors is None else float(avg_neighbors)
        self.auto_avg_neighbors = avg_neighbors is None
        self.nonlinearity = nonlinearity
        self.gate_activation = gate_activation
        self.gate_bias = gate_bias
        self.layer_norm = layer_norm
        self.element_agnostic_interaction = element_agnostic_interaction
        self.element_agnostic_product = element_agnostic_product
        self.n_element_interaction_features = n_element_interaction_features
        self.recompute_radial = recompute_radial
        self.use_triton = use_triton

        self.node_dim = sum((2 * l + 1) * n_hidden_feats for l in range(l_max_hidden_feats + 1))

        self.embedding = nn.Embedding(len(elements), n_hidden_feats)
        self.radial_basis = ChebyshevBasis(n_radial, r_max=r_max, r_min=r_min, mapping=radial_mapping)
        self.cutoff_fn = PolynomialCutoff(r_max, poly_order=cutoff_poly_order)
        self.irreps = o3.Irreps(l_max_edge_attrs, use_triton=use_triton)
        product_out_l_maxes = [
            l_max_hidden_feats if layer_idx < n_interactions - 1 else 0 for layer_idx in range(n_interactions)
        ]
        interaction_in_l_maxes = [0 if layer_idx == 0 else l_max_hidden_feats for layer_idx in range(n_interactions)]
        self.use_products = correlation > 1
        interaction_out_l_maxes = (
            [l_max_edge_attrs for _ in range(n_interactions)] if self.use_products else product_out_l_maxes
        )

        self.interactions = nn.ModuleList(
            [
                InteractionLayer(
                    in1_l_max=interaction_in_l_max,
                    in2_l_max=l_max_edge_attrs,
                    out_l_max=interaction_out_l_max,
                    in1_features=n_hidden_feats,
                    out_features=n_hidden_feats,
                    n_radial=n_radial,
                    hidden_radial=self.hidden_radial,
                    n_elements=None if element_agnostic_interaction else len(elements),
                    element_features=n_element_interaction_features,
                    avg_neighbors=self.avg_neighbors,
                    use_sc=True,
                    use_layer_norm=layer_norm,
                    use_triton=use_triton,
                    recompute_radial=recompute_radial,
                )
                for interaction_in_l_max, interaction_out_l_max in zip(interaction_in_l_maxes, interaction_out_l_maxes)
            ]
        )
        self.products = nn.ModuleList(
            [
                ProductLayer(
                    in_l_max=l_max_edge_attrs,
                    out_l_max=product_out_l_max,
                    in_features=n_hidden_feats,
                    n_elements=None if element_agnostic_product else len(elements),
                    correlation=correlation,
                    path_reduced=path_reduced_product,
                    use_sc=True,
                    use_layer_norm=layer_norm,
                    use_triton=use_triton,
                )
                for product_out_l_max in product_out_l_maxes
            ]
            if self.use_products
            else []
        )
        if self.nonlinearity:
            self.nonlinearities = nn.ModuleList(
                [
                    EquivariantGatedLayer(
                        l_max=interaction_out_l_max,
                        in_features=n_hidden_feats,
                        hidden_features=n_hidden_feats,
                        gate_activation=gate_activation,
                        gate_bias=gate_bias,
                        use_triton=use_triton,
                    )
                    for interaction_out_l_max in interaction_out_l_maxes
                ]
            )

        self.readout = EnergyReadoutLayer(n_hidden_feats, n_interactions, hidden_readout=self.hidden_readout)
        self.scale_shift = ScaleShiftEnergyLayer(len(elements))

    def forward(self, graph: Dict[str, Any]) -> torch.Tensor:
        """Evaluate site energies from a graph dictionary.

        When edge vectors are absent, they are constructed from positions, cell, and
        periodic shifts. For LAMMPS graphs, the supplied edge vectors are used directly,
        and node features are exchanged before each interaction layer after the first.
        Ranks without ghost atoms also take part, since their atoms may be ghosts on
        neighboring ranks.

        For LAMMPS graphs with more than one interaction layer, evaluation and
        differentiation must run inside
        :func:`~flashcart.utils.lammps.lammps_data_slot`. The LAMMPS calculator manages
        this context automatically.

        Args:
            graph (dict): Graph tensors: ``edge_index``, ``batch``, ``atom_types``, and
                either ``positions``/``cell``/``shifts`` (plus optional ``use_shifts``)
                or ``vectors``. LAMMPS graphs set ``lammps_exchange=True``. Their
                ``batch`` covers owned atoms and ``atom_types`` covers owned atoms
                followed by ghosts. The model infers ``nlocal`` and ``ntotal`` from
                these tensor shapes.

        Returns:
            torch.Tensor: Site energies with shape ``(n_atoms,)``. For LAMMPS graphs,
                only the ``nlocal`` owned atoms are included.
        """
        if not isinstance(graph, Mapping):
            raise TypeError(f"{self.__class__.__name__}.forward expects a tensor dictionary. Use predict() for Data.")

        edge_index = graph["edge_index"]
        batch = graph["batch"]
        atom_types = graph["atom_types"]
        is_lammps = bool(graph.get("lammps_exchange", False))
        if is_lammps:
            nlocal = batch.shape[0]
            ntotal = atom_types.shape[0]
            local_atom_types = atom_types[:nlocal]
        else:
            nlocal = ntotal = None
            local_atom_types = atom_types

        vectors = graph.get("vectors")
        if vectors is None:
            positions = graph["positions"]
            cell = graph["cell"]
            shifts = graph["shifts"]
            use_shifts = graph.get("use_shifts", True)
            if not isinstance(use_shifts, bool):
                use_shifts = uses_periodic_shifts(use_shifts)
            vectors = get_edge_vectors(
                positions,
                cell,
                shifts,
                edge_index,
                batch,
                use_shifts=use_shifts,
            )
        distances = vectors.norm(dim=-1)
        directions = vectors / distances.clamp_min(MIN_EDGE_LENGTH).unsqueeze(-1)
        edge_attrs = self.irreps.forward_normalized(directions)
        edge_feats = self.radial_basis(distances)
        envelope = self.cutoff_fn(distances)

        node_feats = self.embedding(atom_types)

        node_outputs = []
        for layer_idx, interaction in enumerate(self.interactions):
            message_feats = node_feats
            if is_lammps and layer_idx > 0:
                # Exchange owned features of shape (nlocal, C) to include ghost rows.
                message_feats = lammps_exchange_features(node_feats, nlocal=nlocal, ntotal=ntotal)
            interaction_feats = interaction(
                message_feats,
                edge_attrs,
                edge_feats,
                edge_index,
                envelope,
                atom_types=atom_types if not self.element_agnostic_interaction else None,
                out_nodes=nlocal,
            )
            if self.nonlinearity:
                interaction_feats = self.nonlinearities[layer_idx](interaction_feats)
            if self.use_products:
                node_feats = self.products[layer_idx](
                    interaction_feats,
                    None if self.element_agnostic_product else local_atom_types,
                )
            else:
                node_feats = interaction_feats
            node_outputs.append(node_feats[:, : self.n_hidden_feats])

        node_energies = self.readout(node_outputs)
        node_energies = self.scale_shift(node_energies, local_atom_types)

        return node_energies

    def _pre_fit_state_tensors(self) -> list[torch.Tensor]:
        tensors = [self.scale_shift.shifts, self.scale_shift.scales]
        for interaction in self.interactions:
            tensors.append(interaction.sqrt_avg_neighbors)
        return tensors

    def broadcast_pre_fit_state(self, fabric: Fabric) -> None:
        """Broadcast the fitted shifts, scales, and neighbor normalization from rank 0.

        For a single process, the method returns without communication.

        Args:
            fabric (Fabric): Launched Fabric handle providing the broadcast.
        """
        if fabric.world_size <= 1:
            return
        with torch.no_grad():
            for tensor in self._pre_fit_state_tensors():
                tensor.copy_(fabric.broadcast(tensor, src=0))
        avg_neighbors = torch.tensor([self.avg_neighbors], dtype=torch.float64)
        avg_neighbors = fabric.broadcast(avg_neighbors, src=0)
        self.avg_neighbors = float(avg_neighbors.item())

    def pre_fit(self, train_loader: DataLoader) -> None:
        """Initialize atomic energy shifts, energy scales, and neighbor normalization.

        Energy scales are set from the root-mean-square reference force component. The
        same scale is assigned to every element. The average neighbor count is estimated
        only when it was not supplied to the constructor.

        Args:
            train_loader (DataLoader): Training data the statistics are read from.
        """
        self.scale_shift.initialize(
            train_loader,
            atomic_shifts=self.atomic_shifts,
            fit_atomic_shifts=self.fit_atomic_shifts,
        )
        if self.auto_avg_neighbors:
            from flashcart.data.statistics import get_avg_neighbors

            avg_neighbors = get_avg_neighbors(train_loader, r_max=self.r_max)
            if avg_neighbors > 0.0:
                self.avg_neighbors = avg_neighbors
                for interaction in self.interactions:
                    interaction.set_avg_neighbors(avg_neighbors)

    def non_decayable_parameters(self) -> Iterator[nn.Parameter]:
        """Child-layer hooks and the element embedding (no weight decay).

        Yields:
            nn.Parameter: Parameter excluded from weight decay.
        """
        yield from super().non_decayable_parameters()
        yield from self.embedding.parameters()

    def non_muon_parameters(self) -> Iterator[nn.Parameter]:
        """Child-layer hooks and the element embedding (kept out of Muon).

        Yields:
            nn.Parameter: Parameter excluded from Muon updates.
        """
        yield from super().non_muon_parameters()
        yield from self.embedding.parameters()

    def to_model_config(self) -> Dict[str, Any]:
        """Return the configuration needed to reconstruct the model architecture.

        The configuration excludes ``recompute_radial``, which controls memory use
        during a run. Model weights are saved separately.

        Returns:
            dict[str, Any]: Architecture and normalization settings used to reconstruct
                the model.
        """
        return {
            "elements": list(self.elements),
            "r_max": self.r_max,
            "r_min": self.r_min,
            "n_hidden_feats": self.n_hidden_feats,
            "l_max_hidden_feats": self.l_max_hidden_feats,
            "l_max_edge_attrs": self.l_max_edge_attrs,
            "n_radial": self.n_radial,
            "n_interactions": self.n_interactions,
            "correlation": self.correlation,
            "path_reduced_product": self.path_reduced_product,
            "element_agnostic_interaction": self.element_agnostic_interaction,
            "element_agnostic_product": self.element_agnostic_product,
            "n_element_interaction_features": self.n_element_interaction_features,
            "nonlinearity": self.nonlinearity,
            "gate_activation": self.gate_activation,
            "gate_bias": self.gate_bias,
            "layer_norm": self.layer_norm,
            "hidden_radial": list(self.hidden_radial),
            "hidden_readout": list(self.hidden_readout),
            "radial_mapping": self.radial_mapping,
            "cutoff_poly_order": self.cutoff_poly_order,
            "atomic_shifts": self.atomic_shifts,
            "fit_atomic_shifts": self.fit_atomic_shifts,
            "avg_neighbors": float(self.avg_neighbors),
            "use_triton": self.use_triton,
        }

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "FlashCartPotential":
        """Construct a potential from the supplied settings.

        This method does not load the packaged training defaults.

        Args:
            cfg (dict): Constructor settings. ``elements`` and ``r_max`` are required.
                Omitted settings use the Python constructor defaults.

        Returns:
            FlashCartPotential: Newly constructed potential.
        """
        return cls(
            elements=cfg["elements"],
            r_max=cfg["r_max"],
            r_min=cfg.get("r_min", 0.0),
            n_hidden_feats=cfg.get("n_hidden_feats", 32),
            l_max_hidden_feats=cfg.get("l_max_hidden_feats", 1),
            l_max_edge_attrs=cfg.get("l_max_edge_attrs", 3),
            n_radial=cfg.get("n_radial", 8),
            n_interactions=cfg.get("n_interactions", 2),
            correlation=cfg.get("correlation", 3),
            path_reduced_product=cfg.get("path_reduced_product", True),
            element_agnostic_interaction=cfg.get("element_agnostic_interaction", True),
            element_agnostic_product=cfg.get("element_agnostic_product", True),
            n_element_interaction_features=cfg.get("n_element_interaction_features", 16),
            nonlinearity=cfg.get("nonlinearity", False),
            gate_activation=cfg.get("gate_activation", "silu"),
            gate_bias=cfg.get("gate_bias", True),
            layer_norm=cfg.get("layer_norm", False),
            hidden_radial=cfg.get("hidden_radial", [64, 64]),
            hidden_readout=cfg.get("hidden_readout", [64, 64]),
            radial_mapping=cfg.get("radial_mapping", "affine"),
            cutoff_poly_order=cfg.get("cutoff_poly_order", 6),
            atomic_shifts=cfg.get("atomic_shifts"),
            fit_atomic_shifts=cfg.get("fit_atomic_shifts", True),
            avg_neighbors=cfg.get("avg_neighbors"),
            use_triton=cfg.get("use_triton", False),
            recompute_radial=cfg.get("recompute_radial", False),
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"elements={self.elements}, r_min={self.r_min}, r_max={self.r_max}, "
            f"n_hidden_feats={self.n_hidden_feats}, l_max_hidden_feats={self.l_max_hidden_feats}, "
            f"l_max_edge_attrs={self.l_max_edge_attrs}, n_radial={self.n_radial}, "
            f"n_interactions={self.n_interactions}, correlation={self.correlation}, "
            f"path_reduced_product={self.path_reduced_product}, "
            f"element_agnostic_interaction={self.element_agnostic_interaction}, "
            f"element_agnostic_product={self.element_agnostic_product}, "
            f"nonlinearity={self.nonlinearity!r}, "
            f"gate_activation={self.gate_activation!r}, "
            f"gate_bias={self.gate_bias}, "
            f"layer_norm={self.layer_norm}, "
            f"radial_mapping={self.radial_mapping!r}, "
            f"avg_neighbors={self.avg_neighbors}, "
            f"atomic_shifts={self.atomic_shifts}, fit_atomic_shifts={self.fit_atomic_shifts}, "
            f"hidden_radial={self.hidden_radial}, hidden_readout={self.hidden_readout}, "
            f"cutoff_poly_order={self.cutoff_poly_order})"
        )
