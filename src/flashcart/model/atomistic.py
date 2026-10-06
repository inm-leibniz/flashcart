from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union

import torch
import torch.nn as nn

from flashcart.utils.config import load_yaml, save_yaml
from flashcart.utils.geometry import to_voigt6, uses_periodic_shifts
from flashcart.utils.parameter_groups import iter_child_special_parameters
from flashcart.utils.scatter import scatter_sum
from flashcart.utils.torch_geometric.data import Data
from flashcart.utils.torch_geometric.dataloader import DataLoader


def save_inference_metadata(
    ckpt_dir: Path,
    model_config: Dict[str, Any],
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Write the model configuration and optional metadata to a checkpoint directory.

    Args:
        ckpt_dir (Path): Checkpoint directory.
        model_config (dict): Constructor config as returned by ``to_model_config``.
        meta (dict, optional): Additional metadata written to ``meta.yaml``. Default:
            None.
    """
    ckpt_dir = Path(ckpt_dir)
    save_yaml(ckpt_dir / "model_config.yaml", model_config)
    if meta is not None:
        save_yaml(ckpt_dir / "meta.yaml", meta)


def load_model_config(ckpt_dir: Path) -> Dict[str, Any]:
    """Read model_config.yaml from a checkpoint directory (raises if absent).

    Args:
        ckpt_dir (Path): Checkpoint directory.

    Returns:
        dict[str, Any]: Model constructor settings read from ``model_config.yaml``.
    """
    ckpt_dir = Path(ckpt_dir)
    model_cfg_path = ckpt_dir / "model_config.yaml"
    if not model_cfg_path.exists():
        raise FileNotFoundError(f"No model_config.yaml in {ckpt_dir}.")
    return load_yaml(model_cfg_path)


class AtomisticModel(nn.Module):
    """Base class for potentials that map atomic graphs to site energies.

    Subclasses implement ``forward`` and the methods ``from_config`` and
    ``to_model_config``. This class sums site energies over each structure and obtains
    forces, stress, and edge forces by automatic differentiation. It also provides
    optional compilation, saving and loading checkpoints, and parameter selection
    for optimizers.
    """

    def __init__(self):
        super().__init__()
        self._compiled_predict_cache: Dict[tuple[bool, bool, bool, bool, str, bool, bool], Any] = {}

    def forward(self, graph: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return per-atom energies for a graph given as a tensor dict.

        Args:
            graph (dict): Graph tensors (see FlashCartPotential.forward).

        Returns:
            torch.Tensor: Site energies with shape ``(n_atoms,)``.
        """
        raise NotImplementedError()

    def predict(
        self,
        graph: Data,
        compute_forces: bool = False,
        compute_stress: bool = False,
        compute_edge_forces: bool = False,
        create_graph: bool = False,
        use_compile: bool = False,
        compile_mode: str = "reduce-overhead",
        fullgraph: bool = True,
        dynamic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Predict energies and requested derivatives for a batched graph.

        Forces are ``-dE/dpositions``. Stress comes from a symmetrized virtual cell
        strain and is returned in Voigt-6 form together with
        ``virials = -stress * volume``. Edge forces are ``dE/dvectors`` for graphs
        carrying precomputed edge vectors (LAMMPS). When derivatives are requested
        without ``create_graph``, the returned energies are detached.

        Gradient recording must remain enabled when requesting energy derivatives.
        Calling ``eval()`` is compatible with derivative prediction, but
        ``torch.no_grad()`` and ``torch.inference_mode()`` are not.

        Position forces and stress require energies computed from positions and cell
        data. For graphs with precomputed ``vectors``, request edge forces instead: the
        supplied vectors are not reconstructed from positions or transformed by the
        virtual strain.

        Args:
            graph (Data): Batched atomic graph.
            compute_forces (bool, optional): Return ``forces``. Default: False.
            compute_stress (bool, optional): Return ``stress`` and ``virials``.
                Requires a nonsingular cell for every structure. Default: False.
            compute_edge_forces (bool, optional): Return ``edge_forces``. Requires
                ``graph.vectors``. Default: False.
            create_graph (bool, optional): Construct a differentiable graph for the
                energy derivatives, allowing training on forces or stress. Default:
                False.
            use_compile (bool, optional): Evaluate a compiled implementation, cached by
                the prediction and compilation options. Default: False.
            compile_mode (str, optional): ``torch.compile`` mode. Default:
                "reduce-overhead".
            fullgraph (bool, optional): Compile without graph breaks. Default: True.
            dynamic (bool, optional): Allow dynamic shapes when compiling. Default:
                False.

        Returns:
            dict[str, torch.Tensor]: For ordinary batched graphs, ``node_energies`` has
                shape ``(n_atoms,)`` and ``energy`` has shape ``(n_graphs,)``. Requested
                derivatives are returned as ``forces`` with shape ``(n_atoms, 3)``,
                ``stress`` and ``virials`` with shape ``(n_graphs, 6)``, and
                ``edge_forces`` with shape ``(n_edges, 3)``. Stress and virials use the
                order ``xx, yy, zz, yz, xz, xy``. For LAMMPS graphs, site energies cover
                only the owned atoms.
        """
        graph_dict = self._prepare_predict_graph(
            graph,
            compute_forces=compute_forces,
            compute_stress=compute_stress,
            compute_edge_forces=compute_edge_forces,
        )

        if use_compile:
            predict_fn = self._compiled_predict_impl(
                compute_forces=compute_forces,
                compute_stress=compute_stress,
                compute_edge_forces=compute_edge_forces,
                create_graph=create_graph,
                compile_mode=compile_mode,
                fullgraph=fullgraph,
                dynamic=dynamic,
            )
            return predict_fn(graph_dict)

        return self._predict_impl(
            graph_dict,
            compute_forces=compute_forces,
            compute_stress=compute_stress,
            compute_edge_forces=compute_edge_forces,
            create_graph=create_graph,
        )

    def _prepare_predict_graph(
        self,
        graph: Data,
        compute_forces: bool,
        compute_stress: bool,
        compute_edge_forces: bool,
    ) -> Dict[str, torch.Tensor]:
        """Prepare independent geometry tensors for the requested derivatives.

        Detach the supplied positions and cell from earlier derivative graphs.
        Stress requests introduce a symmetric strain, while force requests enable
        gradients on positions or supplied edge vectors. The input graph is not
        modified.

        Args:
            graph (Data): Atomic graph with geometry and connectivity.
            compute_forces (bool): Enable position derivatives.
            compute_stress (bool): Introduce a differentiable cell strain.
            compute_edge_forces (bool): Enable derivatives of supplied edge vectors.

        Returns:
            dict[str, torch.Tensor]: Graph fields prepared for energy evaluation and
                the requested derivatives.
        """
        graph_dict = graph.to_dict()
        graph_dict["use_shifts"] = uses_periodic_shifts(graph_dict.get("use_shifts", graph_dict.get("pbc")))

        requires_grad = compute_forces or compute_stress or compute_edge_forces
        if not requires_grad:
            graph_positions = graph_dict["positions"]
            graph_cell = graph_dict.get("cell")
            positions = graph_positions.detach() if graph_positions.requires_grad else graph_positions
            cell = graph_cell.detach() if torch.is_tensor(graph_cell) and graph_cell.requires_grad else graph_cell
            if positions is graph_positions and cell is graph_cell:
                return graph_dict
            graph_dict = dict(graph_dict)
            graph_dict["positions"] = positions
            graph_dict["cell"] = cell
            return graph_dict

        positions = graph_dict["positions"].detach()
        cell = graph_dict["cell"].detach() if torch.is_tensor(graph_dict.get("cell")) else None

        displacement: Optional[torch.Tensor] = None
        if compute_stress:
            if graph_dict.get("cell") is None:
                raise ValueError("Cannot compute stress without cell information.")
            displacement = torch.zeros_like(cell)
            displacement.requires_grad_(True)
            sym_disp = 0.5 * (displacement + displacement.transpose(-1, -2))
            scaling = torch.eye(3, device=cell.device, dtype=cell.dtype).unsqueeze(0) + sym_disp
            scaling_per_atom = scaling[graph_dict["batch"]]
            positions = (positions.unsqueeze(-2) @ scaling_per_atom).squeeze(-2)
            cell = cell @ scaling

        if compute_forces:
            positions.requires_grad_(True)

        graph_dict = dict(graph_dict)
        graph_dict["positions"] = positions
        graph_dict["cell"] = cell
        if compute_edge_forces:
            vectors = graph_dict.get("vectors")
            if vectors is None:
                raise ValueError("Cannot compute edge forces without graph.vectors.")
            vectors = vectors.detach()
            vectors.requires_grad_(True)
            graph_dict["vectors"] = vectors
        if displacement is not None:
            graph_dict["displacement"] = displacement
        return graph_dict

    @staticmethod
    def _energy_grads(
        energy: torch.Tensor,
        inputs: Any,
        graph_dict: Dict[str, torch.Tensor],
        create_graph: bool,
        retain_graph: bool,
        allow_unused: bool = False,
    ) -> Any:
        """Differentiate graph energies with optional weights for padded graphs.

        Without a mask, all graph energies contribute equally. A ``graph_mask``
        supplies one derivative weight per graph, excluding padding graphs when
        their weights are zero.

        Args:
            energy (torch.Tensor): Total energies with shape ``(n_graphs,)``.
            inputs (torch.Tensor | Sequence[torch.Tensor]): Differentiation inputs.
            graph_dict (dict): Graph fields, optionally including ``graph_mask``.
            create_graph (bool): Keep the returned gradients differentiable.
            retain_graph (bool): Keep the graph for further derivative calls.
            allow_unused (bool, optional): Allow inputs absent from the energy
                computation. Default: False.

        Returns:
            tuple[torch.Tensor | None, ...]: One gradient per input. An unused input
                produces None when ``allow_unused`` is True.
        """
        mask = graph_dict.get("graph_mask")
        if mask is None:
            return torch.autograd.grad(
                energy.sum(),
                inputs,
                create_graph=create_graph,
                retain_graph=retain_graph,
                allow_unused=allow_unused,
            )
        return torch.autograd.grad(
            energy,
            inputs,
            grad_outputs=mask,
            create_graph=create_graph,
            retain_graph=retain_graph,
            allow_unused=allow_unused,
        )

    def _predict_impl(
        self,
        graph_dict: Dict[str, torch.Tensor],
        compute_forces: bool,
        compute_stress: bool,
        compute_edge_forces: bool,
        create_graph: bool,
    ) -> Dict[str, torch.Tensor]:
        node_energies = self(graph_dict)
        out = {"node_energies": node_energies}

        n_graphs = int(graph_dict["n_atoms"].shape[0])
        out["energy"] = scatter_sum(out["node_energies"], graph_dict["batch"], dim=0, dim_size=n_graphs)

        if compute_forces and compute_stress:
            dE_dR, dE_dD = self._energy_grads(
                out["energy"],
                [graph_dict["positions"], graph_dict["displacement"]],
                graph_dict,
                create_graph=create_graph,
                retain_graph=create_graph or compute_edge_forces,
            )
            out["forces"] = -dE_dR
            cell_volume = torch.det(graph_dict["cell"])
            stress_3x3 = dE_dD / cell_volume.view(-1, 1, 1)
            out["stress"] = to_voigt6(stress_3x3)
            out["virials"] = -out["stress"] * cell_volume.unsqueeze(-1)
        elif compute_forces:
            (dE_dR,) = self._energy_grads(
                out["energy"],
                graph_dict["positions"],
                graph_dict,
                create_graph=create_graph,
                retain_graph=create_graph or compute_edge_forces,
            )
            out["forces"] = -dE_dR
        elif compute_stress:
            (dE_dD,) = self._energy_grads(
                out["energy"],
                graph_dict["displacement"],
                graph_dict,
                create_graph=create_graph,
                retain_graph=create_graph or compute_edge_forces,
            )
            cell_volume = torch.det(graph_dict["cell"])
            stress_3x3 = dE_dD / cell_volume.view(-1, 1, 1)
            out["stress"] = to_voigt6(stress_3x3)
            out["virials"] = -out["stress"] * cell_volume.unsqueeze(-1)

        if compute_edge_forces:
            (dE_dV,) = self._energy_grads(
                out["energy"],
                graph_dict["vectors"],
                graph_dict,
                create_graph=create_graph,
                retain_graph=create_graph,
                allow_unused=True,
            )
            out["edge_forces"] = dE_dV if dE_dV is not None else torch.zeros_like(graph_dict["vectors"])

        if (compute_forces or compute_stress or compute_edge_forces) and not create_graph:
            out["node_energies"] = out["node_energies"].detach()
            out["energy"] = out["energy"].detach()

        return out

    def _compiled_predict_impl(
        self,
        compute_forces: bool,
        compute_stress: bool,
        compute_edge_forces: bool,
        create_graph: bool,
        compile_mode: str,
        fullgraph: bool,
        dynamic: bool,
    ) -> Any:
        key = (
            compute_forces,
            compute_stress,
            compute_edge_forces,
            create_graph,
            compile_mode,
            fullgraph,
            dynamic,
        )
        if key not in self._compiled_predict_cache:
            if compute_forces or compute_stress or compute_edge_forces:
                from flashcart.utils.compile import configure_autograd_for_compile

                configure_autograd_for_compile(allow_autograd=True)

            def predict_fn(graph_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
                return self._predict_impl(
                    graph_dict,
                    compute_forces=compute_forces,
                    compute_stress=compute_stress,
                    compute_edge_forces=compute_edge_forces,
                    create_graph=create_graph,
                )

            self._compiled_predict_cache[key] = torch.compile(
                predict_fn,
                mode=compile_mode,
                fullgraph=fullgraph,
                dynamic=dynamic,
            )
        return self._compiled_predict_cache[key]

    def clear_compile_cache(self) -> None:
        """Clear cached prediction functions so they recompile on next use."""
        self._compiled_predict_cache.clear()

    def _apply(self, fn):
        self.clear_compile_cache()
        return super()._apply(fn)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_compiled_predict_cache"] = {}
        return state

    def non_decayable_parameters(self) -> Iterator[nn.Parameter]:
        """Parameters excluded from weight decay, aggregated from child layers.

        Yields:
            nn.Parameter: Parameter excluded from weight decay.
        """
        yield from iter_child_special_parameters(self, "non_decayable_parameters")

    def non_muon_parameters(self) -> Iterator[nn.Parameter]:
        """Yield child-layer parameters routed to the fallback optimizer.

        Yields:
            nn.Parameter: Parameter excluded from Muon updates.
        """
        yield from iter_child_special_parameters(self, "non_muon_parameters")

    def pre_fit(self, train_loader: DataLoader) -> None:
        """Hook for fitting data statistics before training (default: no-op).

        Args:
            train_loader (DataLoader): Training data.
        """
        pass

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "AtomisticModel":
        """Construct a model from the subclass's constructor settings.

        Args:
            cfg (dict): Keys as accepted by the subclass constructor.

        Returns:
            AtomisticModel: Model constructed from the supplied settings.
        """
        raise NotImplementedError(f"{cls.__name__} does not implement from_config().")

    def to_model_config(self) -> Dict[str, Any]:
        """Return the constructor settings needed to reconstruct the model.

        Returns:
            dict[str, Any]: Model constructor settings, excluding trained weights.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement to_model_config().")

    def save_weights(self, path: Union[str, Path], filename: str = "model.pt") -> None:
        """Save the state dict.

        Args:
            path (str | Path): Directory, created if missing.
            filename (str, optional): Weights file name. Default: "model.pt".
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / filename)

    def load_weights(
        self,
        path: Union[str, Path],
        filename: str = "model.pt",
        map_location: Union[str, torch.device] = "cpu",
    ) -> None:
        """Load a state dict saved by ``save_weights``.

        Args:
            path (str | Path): Directory containing the weights file.
            filename (str, optional): Weights file name. Default: "model.pt".
            map_location (str | torch.device, optional): Where tensors are loaded.
                Default: "cpu".
        """
        state = torch.load(Path(path) / filename, weights_only=True, map_location=map_location)
        self.load_state_dict(state)

    def save_inference_checkpoint(
        self,
        path: Union[str, Path],
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save the weights and configuration needed to reconstruct the model.

        The checkpoint contains ``model.pt`` and ``model_config.yaml``, with optional
        run metadata in ``meta.yaml``.

        Args:
            path (str | Path): Checkpoint directory, created if missing.
            meta (dict, optional): Additional metadata written to ``meta.yaml``.
                Default: None.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.save_weights(path)
        save_inference_metadata(path, self.to_model_config(), meta=meta)

    @classmethod
    def load_model_config(cls, ckpt_dir: Union[str, Path]) -> Dict[str, Any]:
        """Read model_config.yaml from a checkpoint directory.

        Args:
            ckpt_dir (str | Path): Checkpoint directory.

        Returns:
            dict[str, Any]: Model constructor settings read from the checkpoint.
        """
        return load_model_config(Path(ckpt_dir))

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_dir: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        map_location: Union[str, torch.device] = "cpu",
    ) -> "AtomisticModel":
        """Rebuild a model from a checkpoint directory.

        Args:
            ckpt_dir (str | Path): Directory with model_config.yaml and model.pt.
            device (str | torch.device, optional): Device to which the reconstructed
                model is moved. Default: None, retaining the device used by the
                constructor.
            map_location (str | torch.device, optional): Load location for the state
                dict. Default: "cpu".

        Returns:
            AtomisticModel: Reconstructed model with loaded weights in evaluation mode.
        """
        ckpt_dir = Path(ckpt_dir)
        model = cls.from_config(cls.load_model_config(ckpt_dir))
        model.load_weights(ckpt_dir, map_location=map_location)
        if device is not None:
            model = model.to(device)
        model.eval()
        return model
