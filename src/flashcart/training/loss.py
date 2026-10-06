from typing import Any, Dict, List, Optional, Literal, Union, get_args

import torch
import torch.nn.functional as F

from flashcart.utils.torch_geometric.data import Data

Reduce = Literal["sae", "mae", "sse", "mse", "rmse", "maxe", "huber_sum", "huber_mean", "norm_sum", "norm_mean"]
Normalization = Literal["per_atom", "per_sqrt_atom", "none"]
PropertyKey = Literal["energy", "forces", "stress", "virials"]


def merge_compute_keys(losses: List["LossFunction"]) -> Dict[str, bool]:
    """Combine the prediction requirements of several losses.

    Args:
        losses (list[LossFunction]): Losses to merge.

    Returns:
        dict[str, bool]: Combined force, stress, and virial requirements.
    """
    result = {"compute_forces": False, "compute_stress": False, "compute_virials": False}
    for lf in losses:
        for k, v in lf.compute_keys.items():
            result[k] = result[k] or v
    return result


class LossFunction:
    """Interface for training losses and evaluation metrics.

    ``reduce_batch`` computes the statistics needed from one batch, and
    ``reduce_overall`` combines them using the total atom and structure counts. Means
    and square roots are applied after aggregation so that their normalization does not
    depend on batch size.
    """

    @property
    def name(self) -> str:
        """Column name in logs and metrics.csv, e.g. ``"forces_mae"``."""
        raise NotImplementedError()

    @property
    def compute_keys(self) -> Dict[str, bool]:
        """Prediction requirements used to select energy derivatives.

        The mapping contains ``compute_forces``, ``compute_stress``, and
        ``compute_virials``. Training tasks combine stress and virial requests into the
        model's ``compute_stress`` flag.

        Returns:
            dict[str, bool]: Required force, stress, and virial calculations.
        """
        return {"compute_forces": False, "compute_stress": False, "compute_virials": False}

    @property
    def is_extensive(self) -> bool:
        """True for sum-reduced losses, which scale with the batch size."""
        return False

    def __call__(self, results: Dict[str, torch.Tensor], batch: Data) -> torch.Tensor:
        """Evaluate the loss for one batch.

        Args:
            results (dict[str, torch.Tensor]): Model predictions.
            batch (Data): Batch containing reference targets.

        Returns:
            torch.Tensor: Scalar loss or metric.
        """
        raise NotImplementedError()

    def training_loss(
        self,
        results: Dict[str, torch.Tensor],
        batch: Data,
        world_size: int = 1,
    ) -> torch.Tensor:
        """Evaluate the per-rank loss for distributed gradient computation.

        Extensive (sum-reduced) losses are multiplied by the world size: DDP averages
        gradients across ranks, and the rescale turns that average back into the
        global-batch sum.

        Args:
            results (dict): Model predictions.
            batch (Data): Batch with reference targets.
            world_size (int, optional): Number of DDP ranks. Default: 1.

        Returns:
            torch.Tensor: Scalar loss, scaled for distributed gradient averaging when
                the reduction is extensive.
        """
        loss = self(results, batch)
        if world_size > 1 and self.is_extensive:
            return loss * world_size
        return loss

    def reduce_batch(self, results: Dict[str, torch.Tensor], batch: Data) -> Dict[str, torch.Tensor]:
        """Compute batch statistics to be combined by ``reduce_overall``.

        Args:
            results (dict): Model predictions.
            batch (Data): Batch with reference targets.

        Returns:
            dict[str, torch.Tensor]: Batch statistics keyed by metric name.
        """
        raise NotImplementedError()

    def reduce_overall(self, losses: List[Dict[str, torch.Tensor]], n_atoms: int, n_structures: int) -> torch.Tensor:
        """Compute the metric from accumulated batch statistics.

        Args:
            losses (list[dict]): Batch statistics from ``reduce_batch``. Distributed
                accumulators combine maxima with ``max`` and other statistics with
                ``sum`` before calling this method.
            n_atoms (int): Total number of atoms seen.
            n_structures (int): Total number of structures seen.

        Returns:
            torch.Tensor: Scalar metric over the accumulated batches.
        """
        raise NotImplementedError()


class PropertyLossFunction(LossFunction):
    """Evaluate a loss function or metric for one predicted property.

    The reduction modes ``sae``, ``sse``, ``huber_sum``, and ``norm_sum`` are extensive
    sums. ``mae``, ``mse``, ``huber_mean``, and ``norm_mean`` are per-component or
    per-vector means. ``rmse`` is the root mean square error, and ``maxe`` is the
    maximum absolute error. For vector-valued residuals, ``norm_sum`` and ``norm_mean``
    reduce ``sqrt(sum(residual**2) + 1e-12)`` along the final dimension. For scalar
    energies, they reduce absolute residuals.

    Args:
        key (str): Property: "energy", "forces", "stress", or "virials".
        reduce (str, optional): Reduction mode (see above). Default: "mae".
        normalization (str, optional): Residual normalization, "per_atom",
            "per_sqrt_atom", or "none". Only "none" is valid for the
            per-atom/already-normalized properties (forces, stress). Default:
            "per_atom".
        delta (float, optional): Positive Huber transition point in the units of
            the normalized residual. Required for Huber reductions and rejected for
            other reductions. Default: None.
    """

    def __init__(
        self,
        key: PropertyKey,
        reduce: Reduce = "mae",
        normalization: Normalization = "per_atom",
        delta: Optional[float] = None,
    ):
        if key in ["forces", "stress"] and normalization != "none":
            raise ValueError(
                f"{normalization=} is not supported for {key=} since it is either defined per atom "
                "(e.g., forces are defined per atom) or is already normalized "
                "(e.g., stresses are normalized by the volume)."
            )
        if reduce in ("huber_sum", "huber_mean"):
            if delta is None or delta <= 0.0:
                raise ValueError(f"delta must be positive when {reduce=}. Provided: {delta}.")
        elif delta is not None:
            raise ValueError(
                f"delta is only supported for reduce='huber_sum' or reduce='huber_mean'. Provided {reduce=}."
            )
        self.key = key
        self.reduce = reduce
        self.normalization = normalization
        self.delta = delta

    @property
    def name(self) -> str:
        if self.normalization != "none":
            return f"{self.key}_{self.normalization}_{self.reduce}"
        return f"{self.key}_{self.reduce}"

    def get_residuals(self, results: Dict[str, torch.Tensor], batch: Data) -> torch.Tensor:
        """Normalized prediction-minus-target residuals.

        Args:
            results (dict): Model predictions.
            batch (Data): Batch with reference targets.

        Returns:
            torch.Tensor: Normalized prediction-minus-target residuals with the
                predicted property's shape.
        """
        pred = results.get(self.key)
        target = getattr(batch, self.key, None)
        if pred is None:
            raise ValueError(f"Missing predictions for key {self.key!r}.")
        if target is None:
            raise ValueError(f"Missing target for key {self.key!r}.")

        res = pred - target

        if self.normalization == "none":
            return res
        if self.normalization == "per_atom":
            n_atoms = batch.n_atoms.to(dtype=res.dtype, device=res.device)
            while n_atoms.dim() < res.dim():
                n_atoms = n_atoms.unsqueeze(-1)
            return res / n_atoms
        if self.normalization == "per_sqrt_atom":
            n_atoms_sqrt = batch.n_atoms.to(dtype=res.dtype, device=res.device).sqrt()
            while n_atoms_sqrt.dim() < res.dim():
                n_atoms_sqrt = n_atoms_sqrt.unsqueeze(-1)
            return res / n_atoms_sqrt
        raise RuntimeError(
            f"normalization={self.normalization} is not supported. " f"Supported values: {get_args(Normalization)}."
        )

    @property
    def compute_keys(self) -> Dict[str, bool]:
        return {
            "compute_forces": self.key == "forces",
            "compute_stress": self.key == "stress",
            "compute_virials": self.key == "virials",
        }

    @property
    def is_extensive(self) -> bool:
        return self.reduce in ("sae", "sse", "huber_sum", "norm_sum")

    def __call__(self, results: Dict[str, torch.Tensor], batch: Data) -> torch.Tensor:
        n_structures = int(batch.n_atoms.shape[0])
        if self.reduce in ("mae", "mse", "rmse", "huber_mean", "norm_mean"):
            n_atoms = int(batch.n_atoms.sum().item())
        else:
            n_atoms = 0
        return self.reduce_overall([self.reduce_batch(results, batch)], n_atoms, n_structures)

    def reduce_batch(self, results: Dict[str, torch.Tensor], batch: Data) -> Dict[str, torch.Tensor]:
        """Compute the batch statistic for this reduction, keyed by ``name``.

        Args:
            results (dict): Model predictions.
            batch (Data): Batch with reference targets.

        Returns:
            dict[str, torch.Tensor]: A scalar sum or maximum keyed by ``name``.
        """
        res = self.get_residuals(results, batch)
        if self.reduce in ["sae", "mae"]:
            return {self.name: res.abs().sum()}
        if self.reduce in ["sse", "mse", "rmse"]:
            return {self.name: res.square().sum()}
        if self.reduce in ["huber_sum", "huber_mean"]:
            huber = F.huber_loss(res, torch.zeros_like(res), reduction="none", delta=self.delta)
            return {self.name: huber.sum()}
        if self.reduce in ["norm_sum", "norm_mean"]:
            if res.dim() > 1:
                per_item = (res.square().sum(dim=-1) + 1.0e-12).sqrt()
            else:
                per_item = res.abs()
            return {self.name: per_item.sum()}
        if self.reduce == "maxe":
            return {self.name: res.abs().max()}
        raise RuntimeError(f"reduce={self.reduce} is not supported. " f"Supported values: {get_args(Reduce)}.")

    def get_denominator(self, n_atoms: int, n_structures: int) -> int:
        """Count the observations used to normalize mean reductions.

        Args:
            n_atoms (int): Total atoms over the accumulated batches.
            n_structures (int): Total structures over the accumulated batches.

        Returns:
            int: Number of scalar components or vectors used to normalize the selected
                reduction.
        """
        if self.reduce in ("norm_sum", "norm_mean"):
            return n_atoms if self.key == "forces" else n_structures
        if self.key == "forces":
            return n_atoms * 3
        if self.key in ["stress", "virials"]:
            return n_structures * 6
        return n_structures

    def reduce_overall(self, losses: List[Dict[str, torch.Tensor]], n_atoms: int, n_structures: int) -> torch.Tensor:
        """Combine accumulated batch statistics into the final metric.

        Args:
            losses (list[dict]): Batch sums or maxima from ``reduce_batch``.
            n_atoms (int): Total atoms represented by the statistics.
            n_structures (int): Total structures represented by the statistics.

        Returns:
            torch.Tensor: Scalar metric over the accumulated batches.
        """
        values = [l[self.name] for l in losses]
        denominator = self.get_denominator(n_atoms, n_structures)
        if self.reduce == "maxe":
            return max(values)
        if self.reduce in ["sae", "sse", "huber_sum", "norm_sum"]:
            return sum(values)
        if self.reduce in ["mae", "mse", "huber_mean", "norm_mean"]:
            return sum(values) / denominator
        if self.reduce == "rmse":
            return (sum(values) / denominator).sqrt()
        raise RuntimeError(f"reduce={self.reduce!r} is not supported. " f"Supported values: {get_args(Reduce)}.")


class WeightedSumLoss(LossFunction):
    """Combine property loss functions with specified weights.

    Component losses must have distinct names when batch statistics are accumulated.
    The name identifies the property, normalization, and reduction, but does not
    include the Huber threshold.

    Args:
        losses (list[PropertyLossFunction]): Component losses.
        weights (list[float], optional): One weight per component. Default: all 1.0.
    """

    def __init__(self, losses: List[PropertyLossFunction], weights: Optional[List[float]] = None):
        if not losses:
            raise ValueError("WeightedSumLoss requires at least one PropertyLossFunction.")
        if weights is not None and len(weights) != len(losses):
            raise ValueError("Weights must be provided for all losses.")
        self.losses = losses
        self.weights = weights if weights is not None else [1.0] * len(losses)

    @property
    def name(self) -> str:
        return "+".join(loss.name for loss in self.losses)

    @property
    def compute_keys(self) -> Dict[str, bool]:
        return merge_compute_keys(self.losses)

    @property
    def is_extensive(self) -> bool:
        return all(loss.is_extensive for loss in self.losses)

    def __call__(self, results: Dict[str, torch.Tensor], batch: Data) -> torch.Tensor:
        return sum(w * loss(results, batch) for loss, w in zip(self.losses, self.weights))

    def training_loss(
        self,
        results: Dict[str, torch.Tensor],
        batch: Data,
        world_size: int = 1,
    ) -> torch.Tensor:
        return sum(w * loss.training_loss(results, batch, world_size) for loss, w in zip(self.losses, self.weights))

    def reduce_batch(self, results: Dict[str, Any], batch: Data) -> Dict[str, torch.Tensor]:
        batch_losses = {}
        for loss in self.losses:
            batch_losses.update(loss.reduce_batch(results, batch))
        return batch_losses

    def reduce_overall(self, losses: List[Dict[str, torch.Tensor]], n_atoms: int, n_structures: int) -> torch.Tensor:
        return sum(w * loss.reduce_overall(losses, n_atoms, n_structures) for loss, w in zip(self.losses, self.weights))


class LossAccumulator:
    """Accumulate batch statistics and counts for a loss or metric.

    ``update`` stores detached statistics on CPU together with atom and structure
    counts. ``compute`` combines these statistics and, when a distributed Fabric
    instance is provided, communicates them across ranks.

    Args:
        loss (LossFunction): Loss defining how batch statistics are combined.
    """

    def __init__(self, loss: LossFunction):
        self._loss = loss
        self._batch_losses: List[Dict[str, torch.Tensor]] = []
        self._n_structures: int = 0
        self._n_atoms: int = 0

    @torch.no_grad()
    def update(self, results: Dict[str, torch.Tensor], batch: Data) -> None:
        """Store one batch's detached statistics and update the observation counts.

        Args:
            results (dict): Model predictions.
            batch (Data): Batch with reference targets.
        """
        batch_loss = {k: v.detach().cpu() for k, v in self._loss.reduce_batch(results, batch).items()}
        self._batch_losses.append(batch_loss)
        self._n_structures += int(batch.n_atoms.shape[0])
        self._n_atoms += int(batch.n_atoms.sum().item())

    def compute(self, fabric=None) -> torch.Tensor:
        """Compute the metric from accumulated batch statistics.

        Args:
            fabric (Fabric, optional): Fabric instance for distributed reduction.
                Default: None, computing locally.

        Returns:
            torch.Tensor: Scalar metric computed from the accumulated statistics.
        """
        if fabric is None or getattr(fabric, "world_size", 1) == 1:
            return self._loss.reduce_overall(self._batch_losses, self._n_atoms, self._n_structures)

        counts = torch.tensor(
            [self._n_atoms, self._n_structures],
            device=fabric.device,
            dtype=torch.float64,
        )

        global_losses = {}
        keys = self._metric_names()
        for key in keys:
            reduce_op = "max" if key.endswith("_maxe") else "sum"
            value = torch.zeros((), device=fabric.device, dtype=torch.float64)
            if self._batch_losses:
                values = [batch_loss[key] for batch_loss in self._batch_losses]
                local_value = torch.stack(values).amax() if reduce_op == "max" else sum(values)
                value = local_value.to(
                    device=fabric.device,
                    dtype=torch.float64,
                )
            global_losses[key] = fabric.all_reduce(value, reduce_op=reduce_op)
        counts = fabric.all_reduce(counts, reduce_op="sum")
        return self._loss.reduce_overall(
            [global_losses],
            n_atoms=int(counts[0].item()),
            n_structures=int(counts[1].item()),
        )

    def _metric_names(self) -> list[str]:
        if isinstance(self._loss, WeightedSumLoss):
            return [loss.name for loss in self._loss.losses]
        return [self._loss.name]

    def reset(self) -> None:
        """Clear the accumulated partials and counts."""
        self._batch_losses = []
        self._n_structures = 0
        self._n_atoms = 0


def _property_loss_from_dict(spec: Dict[str, Any]) -> PropertyLossFunction:
    delta = spec.get("delta")
    if delta is not None:
        delta = float(delta)
    return PropertyLossFunction(
        spec["property"],
        spec.get("reduce", "mae"),
        spec.get("normalization", "none"),
        delta=delta,
    )


def loss_from_config(spec: Dict[str, Any]) -> LossFunction:
    """Build a property loss or weighted sum from a configuration mapping.

    Args:
        spec (dict): A property specification with ``property`` and optional ``reduce``,
            ``normalization``, and ``delta`` entries, or a ``weighted_sum`` list of
            property specifications with optional ``weight`` entries. Defaults are
            ``reduce="mae"``, ``normalization="none"``, and ``weight=1.0``.

    Returns:
        LossFunction: Configured property loss or weighted sum.
    """
    if "weighted_sum" in spec:
        losses = [_property_loss_from_dict(t) for t in spec["weighted_sum"]]
        weights = [float(t.get("weight", 1.0)) for t in spec["weighted_sum"]]
        return WeightedSumLoss(losses, weights)
    return _property_loss_from_dict(spec)
