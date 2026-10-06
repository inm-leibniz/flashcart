import csv
import logging
import math
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator, List, Optional, Union, Dict

from torch_ema import ExponentialMovingAverage

import torch
from lightning.fabric import Fabric

from flashcart.data.padding import PadAtomicData, slice_padded_outputs
from flashcart.model.atomistic import AtomisticModel
from flashcart.training.loss import LossAccumulator, LossFunction
from flashcart.training.muon_norm import (
    MOMENTUM_EPS_WARN,
    momentum_norm_report,
    muon_weight_decay,
    weight_rms_report,
)
from flashcart.utils.distributed import is_global_zero
from flashcart.utils.torch_geometric.dataloader import DataLoader

log = logging.getLogger("flashcart.train")


@contextmanager
def preserve_grad_state(model: torch.nn.Module) -> Iterator[None]:
    """Freeze all parameters inside the block, restoring their flags on exit.

    Used around evaluation so force/stress autograd tracks only the inputs, not the
    parameters.

    Args:
        model (torch.nn.Module): Model whose parameters are frozen.
    """
    backup = {p: p.requires_grad for p in model.parameters()}
    try:
        for p in backup:
            p.requires_grad_(False)
        yield
    finally:
        for p, req in backup.items():
            p.requires_grad_(req)


def predict_kwargs(losses: List[LossFunction]) -> Dict:
    """Derive the ``predict()`` flags from what the given losses require.

    Args:
        losses (list[LossFunction]): Losses that will consume the predictions.

    Returns:
        dict[str, bool]: ``compute_forces`` and ``compute_stress`` flags required by the
            losses.
    """
    compute_forces = any(lf.compute_keys["compute_forces"] for lf in losses)
    compute_stress = any(lf.compute_keys["compute_stress"] or lf.compute_keys["compute_virials"] for lf in losses)
    return {"compute_forces": compute_forces, "compute_stress": compute_stress}


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Strip Fabric/DDP wrappers down to the underlying model.

    Training forwards must go through the wrapper (gradient synchronization). Use this
    only for evaluation, checkpointing, and attribute access.

    Args:
        model (torch.nn.Module): Possibly wrapped model.

    Returns:
        torch.nn.Module: Underlying module after removing wrappers exposing ``module``.
    """
    while hasattr(model, "module"):
        model = model.module
    return model


def make_predict_padder(
    model: torch.nn.Module, predict_compile: bool, pad_kwargs: Dict[str, int]
) -> Optional[PadAtomicData]:
    """Build a batch padder for compiled prediction.

    Padding reduces tensor-shape changes while batches fit within the allocated
    capacities. Eager prediction runs without padding.

    Args:
        model (torch.nn.Module): Model providing ``r_max`` for fake edges.
        predict_compile (bool): Whether predictions run compiled.
        pad_kwargs (dict): PadAtomicData rounding multiples.

    Returns:
        PadAtomicData | None: Padder for compiled prediction, or None for eager
            prediction.
    """
    if not predict_compile:
        return None
    r_max = float(unwrap_model(model).r_max)
    return PadAtomicData(r_max=r_max, **pad_kwargs)


def seed_padder_from_loader(padder: Optional[PadAtomicData], loader: DataLoader) -> None:
    """Initialize padding capacities from the batch sampler's configured limits.

    These initial capacities reduce growth as batch sizes vary. Batches that exceed them
    may still increase the capacities and trigger recompilation. When sampler limits are
    unavailable, capacities are determined from observed batches.

    Args:
        padder (PadAtomicData, optional): Padder to initialize. Nothing is changed when
            None.
        loader (DataLoader): Loader whose batch sampler carries the maxima.
    """
    if padder is None:
        return
    sampler = getattr(loader, "batch_sampler", None)
    if sampler is None or not hasattr(sampler, "max_nodes"):
        return
    padder.seed_budgets(
        atom_budget=sampler.max_nodes,
        edge_budget=sampler.max_edges,
        graph_budget=sampler.max_graphs,
    )


def padded_predict(
    model: torch.nn.Module,
    batch: Any,
    padder: Optional[PadAtomicData],
    predict_kwargs: Dict[str, Any],
    compile_kwargs: Dict[str, Any],
    create_graph: bool,
) -> Dict[str, torch.Tensor]:
    """Evaluate a batch with optional padding and remove padded output entries.

    Args:
        model (torch.nn.Module): Model exposing ``predict``.
        batch (Any): Batched atomic graph.
        padder (PadAtomicData, optional): None predicts unpadded.
        predict_kwargs (dict): Property flags (see ``predict_kwargs``).
        compile_kwargs (dict): ``use_compile``/mode kwargs, empty for eager.
        create_graph (bool): Keep derivative graphs when training on energy derivatives.

    Returns:
        dict[str, torch.Tensor]: Predictions for the real atoms and structures.
    """
    if padder is None:
        return model.predict(batch, create_graph=create_graph, **predict_kwargs, **compile_kwargs)
    padded, n_real_atoms, n_real_graphs = padder(batch)
    results = model.predict(padded, create_graph=create_graph, **predict_kwargs, **compile_kwargs)
    return slice_padded_outputs(results, n_real_atoms, n_real_graphs)


class TrainingTask:
    """Train a potential with validation and checkpointing.

    Optionally maintain exponential moving averages of the model parameters.

    The ``log`` checkpoint stores the current weights and training state for resumption.
    The ``best`` checkpoint stores the weights that minimize the selected validation
    metric, using averaged weights when enabled. Training continues to ``max_epochs``.
    The validation metric selects checkpoints without applying a patience-based stopping
    rule.

    Args:
        model (torch.nn.Module): Model to train.
        train_loss (LossFunction): Loss backpropagated per step.
        early_stopping_loss (LossFunction): Validation metric selecting the best
            checkpoint.
        eval_losses (list[LossFunction]): Metrics logged for train and valid.
        optimizer (torch.optim.Optimizer): Main optimizer.
        scheduler (LRScheduler, optional): Learning-rate scheduler stepped after each
            epoch. Its first parameter group's learning rate is copied to the auxiliary
            optimizers. Default: None.
        aux_optimizers (list, optional): Additional optimizers stepped after the main
            optimizer. Default: None.
        max_epochs (int, optional): Total epochs. Default: 1000.
        validate_every (int, optional): Epochs between validations. Default: 1.
        save_every (int, optional): Epochs between rolling checkpoints. Default: 100.
        grad_clip_norm (float, optional): Maximum global gradient norm, applied before
            all optimizer steps. This limits gradients rather than directly bounding the
            resulting parameter updates. Default: None.
        log_significant_digits (int, optional): Round CSV values to this many
            significant digits. None logs full precision. Default: 4.
        log_weight_norms (bool, optional): Include Muon weight and momentum diagnostics
            in ``metrics.csv``. Default: False.
        ema_decay (float, optional): Decay factor for the exponential moving average
            used during validation and best-checkpoint selection. Default: None,
            disabling averaging.
        predict_compile (bool, optional): Run predictions through ``torch.compile``
            (enables batch padding). Default: False.
        predict_compile_mode (str, optional): Compile mode. Default: "reduce-overhead".
        predict_compile_fullgraph (bool, optional): Compile without graph breaks.
            Default: True.
        predict_compile_dynamic (bool, optional): Dynamic-shape compile. Default: False.
        predict_pad_atom_multiple (int, optional): Atom-budget rounding of the padder.
            Default: 128.
        predict_pad_edge_multiple (int, optional): Edge-budget rounding. Default: 128.
        predict_pad_graph_multiple (int, optional): Graph-budget rounding. Default: 8.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loss: LossFunction,
        early_stopping_loss: LossFunction,
        eval_losses: List[LossFunction],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        aux_optimizers: Optional[List[torch.optim.Optimizer]] = None,
        max_epochs: int = 1000,
        validate_every: int = 1,
        save_every: int = 100,
        grad_clip_norm: Optional[float] = None,
        log_significant_digits: Optional[int] = 4,
        log_weight_norms: bool = False,
        ema_decay: Optional[float] = None,
        predict_compile: bool = False,
        predict_compile_mode: str = "reduce-overhead",
        predict_compile_fullgraph: bool = True,
        predict_compile_dynamic: bool = False,
        predict_pad_atom_multiple: int = 128,
        predict_pad_edge_multiple: int = 128,
        predict_pad_graph_multiple: int = 8,
    ):
        self.model = model
        self.train_loss = train_loss
        self.early_stopping_loss = early_stopping_loss
        self.eval_losses = eval_losses
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.aux_optimizers: List[torch.optim.Optimizer] = list(aux_optimizers) if aux_optimizers else []
        self.max_epochs = max_epochs
        self.validate_every = validate_every
        self.save_every = save_every
        self.grad_clip_norm = grad_clip_norm
        self.log_significant_digits = log_significant_digits
        self.log_weight_norms = log_weight_norms
        self.ema_decay = ema_decay

        self.predict_compile_kwargs = (
            {
                "use_compile": True,
                "compile_mode": predict_compile_mode,
                "fullgraph": predict_compile_fullgraph,
                "dynamic": predict_compile_dynamic,
            }
            if predict_compile
            else {}
        )

        pad_kwargs = {
            "atom_multiple": predict_pad_atom_multiple,
            "edge_multiple": predict_pad_edge_multiple,
            "graph_multiple": predict_pad_graph_multiple,
        }

        self._train_padder = make_predict_padder(model, predict_compile, pad_kwargs)
        self._eval_padder = make_predict_padder(model, predict_compile, pad_kwargs)

        self.fabric: Optional[Fabric] = None
        self.ema: Optional[ExponentialMovingAverage] = None

        self.epoch: int = 0
        self.best_es_metric: float = float("inf")
        self.best_epoch: int = 0
        self._train_step_idx: int = 0
        self._warned_nonfinite_loss: bool = False
        self._warned_momentum_eps: bool = False
        self._data_nonfinite: Optional[torch.Tensor] = None

        self._csv_path: Optional[Path] = None
        self._csv_columns: Optional[list[str]] = None
        
        self._wd_efoldings: float = 0.0

    def run(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        output_path: Union[str, Path],
        fabric: Optional[Fabric] = None,
    ) -> None:
        """Train through ``max_epochs``, restoring a saved training state when present.

        If ``output_path/log/training_state.pt`` exists, the model weights, optimizer
        states, scheduler, and averaging state are restored before training continues.

        Args:
            train_loader (DataLoader): Training batches.
            valid_loader (DataLoader): Validation batches.
            output_path (str | Path): Run directory (checkpoints, metrics.csv).
            fabric (Fabric, optional): Launched Fabric. A single-process instance is
                created when omitted.
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        if fabric is None:
            fabric = Fabric()
        if not getattr(fabric, "_launched", False):
            fabric.launch()
        self.fabric = fabric

        seed_padder_from_loader(self._train_padder, train_loader)
        seed_padder_from_loader(self._eval_padder, valid_loader)

        loss_names = [loss.name for loss in self.eval_losses]
        self._csv_columns = (
            ["epoch", "epoch_time_s", "lr", "best_es_metric"]
            + [f"train/{n}" for n in loss_names]
            + [f"valid/{n}" for n in loss_names]
        )
        if self.log_weight_norms:
            self._csv_columns += [
                "weight_rms",
                "weight_rms_max",
                "weight_rms_max_tensor",
                "wd_efoldings",
                "momentum_norm_min_over_eps",
            ]
        self._csv_path = output_path / "metrics.csv"
        if is_global_zero(fabric) and not self._csv_path.exists():
            with open(self._csv_path, "w", newline="") as f:
                csv.writer(f).writerow(self._csv_columns)
        if fabric.world_size > 1:
            fabric.barrier()

        result = fabric.setup(self.model, self.optimizer, *self.aux_optimizers, scheduler=self.scheduler)
        self.model = result[0]
        if hasattr(self.model, "mark_forward_method"):
            self.model.mark_forward_method("predict")
        self.optimizer = result[1]
        result_tail = list(result[2:])
        if self.scheduler is not None:
            self.scheduler = result_tail.pop()
        self.aux_optimizers = result_tail
        if self.aux_optimizers and self.optimizer.param_groups:
            lr = self.optimizer.param_groups[0]["lr"]
            for opt in self.aux_optimizers:
                for group in opt.param_groups:
                    group["lr"] = lr
        train_loader, valid_loader = fabric.setup_dataloaders(
            train_loader,
            valid_loader,
            use_distributed_sampler=False,
        )

        if self.ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=self.ema_decay)

        self._data_nonfinite = torch.zeros((), dtype=torch.bool, device=next(self.model.parameters()).device)

        if (output_path / "log" / "training_state.pt").exists():
            unwrap_model(self.model).load_weights(output_path / "log")
            self._load_state(output_path / "log")

        while self.epoch < self.max_epochs:
            start_epoch = time.time()
            self.epoch += 1
            epoch_lr = self.optimizer.param_groups[0]["lr"] if self.optimizer.param_groups else 0.0
            self._set_loader_epoch(train_loader, self.epoch)

            self.model.train()
            train_accs = [LossAccumulator(loss) for loss in self.eval_losses]
            self._train_step_idx = 0
            for batch in train_loader:
                self._train_step(batch, train_accs)

            weight_decay = muon_weight_decay(self.aux_optimizers)
            if weight_decay:
                self._wd_efoldings += 2.0 * weight_decay * epoch_lr * self._train_step_idx

            train_metrics = {
                loss.name: acc.compute(self.fabric).item() for loss, acc in zip(self.eval_losses, train_accs)
            }

            self._stop_if_nonfinite(train_metrics)

            self._step_scheduler()

            if self.epoch % self.save_every == 0:
                self._save_checkpoint(output_path / "log")

            valid_metrics = {}
            es_metric: Optional[float] = None
            if self.epoch % self.validate_every == 0:
                eval_accs = [LossAccumulator(loss) for loss in self.eval_losses]
                es_acc = LossAccumulator(self.early_stopping_loss)
                for batch in valid_loader:
                    self._eval_step(batch, eval_accs + [es_acc])
                valid_metrics = {
                    loss.name: acc.compute(self.fabric).item() for loss, acc in zip(self.eval_losses, eval_accs)
                }
                es_metric = float(es_acc.compute(self.fabric).item())

                if es_metric < self.best_es_metric:
                    self.best_es_metric = es_metric
                    self.best_epoch = self.epoch
                    ema_ctx = self.ema.average_parameters() if self.ema is not None else nullcontext()
                    with ema_ctx:
                        self._save_checkpoint(output_path / "best", ema_weights=self.ema is not None)

            epoch_time = time.time() - start_epoch
            self._log(train_metrics, valid_metrics, epoch_time, lr=epoch_lr)

        self._save_checkpoint(output_path / "log")

    # Call the wrapped model (self.model) to trigger DDP gradient synchronization.
    # training_loss scales extensive losses by world size so averaging gradients
    # across ranks reproduces the global-batch gradient.
    def _train_step(self, batch: Any, accs: List[LossAccumulator]) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        for aux_opt in self.aux_optimizers:
            aux_opt.zero_grad(set_to_none=True)
        train_predict_kwargs = predict_kwargs([self.train_loss])
        results = padded_predict(
            self.model,
            batch,
            self._train_padder,
            train_predict_kwargs,
            self.predict_compile_kwargs,
            create_graph=train_predict_kwargs["compute_forces"] or train_predict_kwargs["compute_stress"],
        )
        world_size = getattr(self.fabric, "world_size", 1)
        total_loss = self.train_loss.training_loss(results, batch, world_size=world_size)
        self._train_step_idx += 1

        for key in ("positions", "energy", "forces"):
            value = getattr(batch, key, None)
            if torch.is_tensor(value) and torch.is_floating_point(value):
                self._data_nonfinite.logical_or_(~torch.isfinite(value).all())

        if not self._warned_nonfinite_loss and self._train_step_idx % 32 == 0 and not bool(torch.isfinite(total_loss)):
            self._warned_nonfinite_loss = True
            log.warning(
                "Non-finite training loss at epoch %d, by step %d (32-step check granularity).",
                self.epoch,
                self._train_step_idx,
            )

        self.fabric.backward(total_loss)
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
        self.optimizer.step()
        for aux_opt in self.aux_optimizers:
            aux_opt.step()
        if self.ema is not None:
            self.ema.update()
        for acc in accs:
            acc.update(results, batch)

    def _stop_if_nonfinite(self, train_metrics: dict[str, float]) -> None:
        bad_grads = sorted(name for opt in self.aux_optimizers for name in (getattr(opt, "nonfinite_report", dict)()))
        bad_metrics = sorted(name for name, value in train_metrics.items() if not math.isfinite(value))
        bad_data = self._data_nonfinite is not None and bool(self._data_nonfinite)
        bad = bool(bad_grads or bad_metrics or bad_data)
        if self.fabric is not None and self.fabric.world_size > 1:
            flag = torch.tensor(float(bad), device=self._data_nonfinite.device)
            bad = float(self.fabric.all_reduce(flag, reduce_op="sum")) > 0.0
        if not bad:
            return
        parts = []
        if bad_grads:
            shown = ", ".join(bad_grads[:4])
            more = f" (and {len(bad_grads) - 4} more)" if len(bad_grads) > 4 else ""
            parts.append(f"gradients in {shown}{more}")
        if bad_metrics:
            parts.append(f"train metrics {', '.join(bad_metrics)}")
        if bad_data:
            parts.append("input data (positions/energy/forces arrived non-finite from the loader)")
        where = "; ".join(parts) if parts else "values on another rank"
        message = (
            f"Non-finite {where} at epoch {self.epoch}. Training stopped rather than continuing on "
            f"poisoned weights; best/ still holds the last checkpoint that improved."
        )
        log.error(message)  # the run's own log, not just the traceback on stderr
        raise RuntimeError(message)

    def _set_loader_epoch(self, loader: DataLoader, epoch: int) -> None:
        for attr in ("batch_sampler", "sampler"):
            sampler = getattr(loader, attr, None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

    # Validation uses EMA weights when enabled and freezes model parameters.
    # No gradient synchronization is needed, so the unwrapped model avoids
    # DDP forward overhead.
    def _eval_step(self, batch: Any, accs: List[LossAccumulator]) -> None:
        ema_ctx = self.ema.average_parameters() if self.ema is not None else nullcontext()
        with ema_ctx, preserve_grad_state(self.model):
            model = unwrap_model(self.model)
            results = padded_predict(
                model,
                batch,
                self._eval_padder,
                predict_kwargs(self.eval_losses + [self.early_stopping_loss]),
                self.predict_compile_kwargs,
                create_graph=False,
            )
            for acc in accs:
                acc.update(results, batch)
            del results

    # The scheduler updates only the main optimizer. Copy its learning rate to
    # the auxiliary optimizers (Muon) so all follow the same schedule.
    def _step_scheduler(self) -> None:
        if self.scheduler is None:
            return
        self.scheduler.step()

        if self.aux_optimizers and self.optimizer.param_groups:
            lr = self.optimizer.param_groups[0]["lr"]
            for opt in self.aux_optimizers:
                for group in opt.param_groups:
                    group["lr"] = lr

    # Rank 0 writes model_<epoch>.pt, the model.pt symlink, inference metadata,
    # and training state. All ranks wait at the barrier until writing finishes.
    def _save_checkpoint(self, ckpt_dir: Path, ema_weights: bool = False) -> None:
        if not is_global_zero(self.fabric):
            if self.fabric is not None:
                self.fabric.barrier()
            return
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        for old in ckpt_dir.glob("model_*.pt"):
            old.unlink(missing_ok=True)
        epoch_file = ckpt_dir / f"model_{self.epoch}.pt"
        model = unwrap_model(self.model)
        torch.save(model.state_dict(), epoch_file)
        link = ckpt_dir / "model.pt"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(epoch_file.name)
        try:
            model_config = model.to_model_config()
        except NotImplementedError:
            model_config = None
        if model_config is not None:
            from flashcart.model.atomistic import save_inference_metadata

            save_inference_metadata(
                ckpt_dir,
                model_config,
                meta={
                    "epoch": self.epoch,
                    "best_epoch": self.best_epoch,
                    "best_es_metric": self.best_es_metric,
                    # With EMA enabled, best/ is saved inside ema.average_parameters()
                    # and contains EMA weights. log/ contains current model weights.
                    "weights": "ema" if ema_weights else "raw",
                },
            )
        self._save_state(ckpt_dir)
        if self.fabric is not None:
            self.fabric.barrier()

    def _save_state(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": self.epoch,
                "best_es_metric": self.best_es_metric,
                "best_epoch": self.best_epoch,
                "wd_efoldings": self._wd_efoldings,
                "optimizer": self.optimizer.state_dict(),
                "aux_optimizers": [opt.state_dict() for opt in self.aux_optimizers],
                "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                "ema": self.ema.state_dict() if self.ema is not None else None,
            },
            path / "training_state.pt",
        )

    def _load_state(self, path: Path) -> None:
        state = torch.load(path / "training_state.pt", weights_only=True, map_location="cpu")
        self.epoch = state["epoch"]
        self.best_es_metric = state["best_es_metric"]
        self.best_epoch = state["best_epoch"]
        self._wd_efoldings = float(state.get("wd_efoldings", 0.0))  # absent before this key existed
        if self.scheduler and state.get("scheduler"):
            self.scheduler.load_state_dict(state["scheduler"])
        self.optimizer.load_state_dict(state["optimizer"])
        for opt, sd in zip(self.aux_optimizers, state.get("aux_optimizers") or []):
            opt.load_state_dict(sd)
        if self.aux_optimizers and self.optimizer.param_groups:
            lr = self.optimizer.param_groups[0]["lr"]
            for opt in self.aux_optimizers:
                for group in opt.param_groups:
                    group["lr"] = lr
        if self.ema is not None and state.get("ema") is not None:
            self.ema.load_state_dict(state["ema"])

    def _log(
        self,
        train_metrics: dict[str, float],
        valid_metrics: dict[str, float],
        epoch_time: float,
        lr: Optional[float] = None,
    ) -> None:
        if not is_global_zero(self.fabric):
            return
        if lr is None:
            lr = self.optimizer.param_groups[0]["lr"]
        row: dict[str, Any] = {
            "epoch": self.epoch,
            "epoch_time_s": epoch_time,
            "lr": lr,
            "best_es_metric": self.best_es_metric,
        }
        momentum_report = momentum_norm_report(self.aux_optimizers)
        ratio = momentum_report.get("momentum_norm_min_over_eps")
        if ratio is not None and ratio < MOMENTUM_EPS_WARN and not self._warned_momentum_eps:
            self._warned_momentum_eps = True
            log.warning(
                "Smallest Muon momentum norm is %.1fx the Newton-Schulz eps clamp at epoch %d. "
                "Below the clamp the update for that tensor silently collapses.",
                ratio,
                self.epoch,
            )
        if self.log_weight_norms:
            row["wd_efoldings"] = self._wd_efoldings
            row.update(weight_rms_report(unwrap_model(self.model), self.aux_optimizers))
            row.update(momentum_report)
        for name, val in train_metrics.items():
            row[f"train/{name}"] = val
        for name, val in valid_metrics.items():
            row[f"valid/{name}"] = val
        row = {key: self._format_log_value(val) for key, val in row.items()}
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_columns, extrasaction="ignore", restval="")
            writer.writerow(row)

    def _format_log_value(self, value: float) -> float:
        if self.log_significant_digits is None:
            return value
        if not isinstance(value, float) or value == 0.0 or not math.isfinite(value):
            return value
        return float(f"{value:.{self.log_significant_digits}g}")


class EvaluationTask:
    """Evaluate a model over a loader and aggregate the requested metrics.

    Args:
        model (torch.nn.Module): Model to evaluate.
        eval_losses (list[LossFunction]): Metrics to compute.
        predict_compile (bool, optional): Compiled predictions (enables batch padding).
            Default: False.
        predict_compile_mode (str, optional): Compile mode. Default: "reduce-overhead".
        predict_compile_fullgraph (bool, optional): Compile without graph breaks.
            Default: True.
        predict_compile_dynamic (bool, optional): Dynamic-shape compile. Default: False.
        predict_pad_atom_multiple (int, optional): Atom-budget rounding of the padder.
            Default: 128.
        predict_pad_edge_multiple (int, optional): Edge-budget rounding. Default: 128.
        predict_pad_graph_multiple (int, optional): Graph-budget rounding. Default: 8.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        eval_losses: List[LossFunction],
        predict_compile: bool = False,
        predict_compile_mode: str = "reduce-overhead",
        predict_compile_fullgraph: bool = True,
        predict_compile_dynamic: bool = False,
        predict_pad_atom_multiple: int = 128,
        predict_pad_edge_multiple: int = 128,
        predict_pad_graph_multiple: int = 8,
    ) -> None:
        self.model = model
        self.eval_losses = eval_losses

        self.predict_compile_kwargs = (
            {
                "use_compile": True,
                "compile_mode": predict_compile_mode,
                "fullgraph": predict_compile_fullgraph,
                "dynamic": predict_compile_dynamic,
            }
            if predict_compile
            else {}
        )

        self._padder = make_predict_padder(
            model,
            predict_compile,
            {
                "atom_multiple": predict_pad_atom_multiple,
                "edge_multiple": predict_pad_edge_multiple,
                "graph_multiple": predict_pad_graph_multiple,
            },
        )

        self.fabric: Optional[Fabric] = None

    def run(
        self,
        data_loader: DataLoader,
        fabric: Optional[Fabric] = None,
    ) -> dict[str, float]:
        """Evaluate over the loader and return {metric name: value}.

        Args:
            data_loader (DataLoader): Batches to evaluate.
            fabric (Fabric, optional): Fabric instance used for batch placement and
                distributed metric reduction. Default: None, retaining any instance
                provided by an earlier call.

        Returns:
            dict[str, float]: Metric values keyed by loss name.
        """
        if fabric is not None:
            self.fabric = fabric

        seed_padder_from_loader(self._padder, data_loader)
        accs = [LossAccumulator(loss) for loss in self.eval_losses]
        device = self.fabric.device if self.fabric is not None else next(self.model.parameters(), torch.empty(0)).device

        with preserve_grad_state(self.model):
            model = unwrap_model(self.model)
            for batch in data_loader:
                if hasattr(batch, "to"):
                    batch = batch.to(device)
                results = padded_predict(
                    model,
                    batch,
                    self._padder,
                    predict_kwargs(self.eval_losses),
                    self.predict_compile_kwargs,
                    create_graph=False,
                )
                for acc in accs:
                    acc.update(results, batch)
                del results

        metrics = {loss.name: acc.compute(self.fabric).item() for loss, acc in zip(self.eval_losses, accs)}
        return metrics
