"""Train a FlashCart potential (the ``flashcart-train`` console script)."""

import argparse
import os
from pathlib import Path

import torch

from flashcart.data.dataset import make_loader
from flashcart.data.utils import get_elements
from flashcart.model.flashcart import FlashCartPotential
from flashcart.training.loss import loss_from_config
from flashcart.training.muon_norm import epoch_learning_rates, forecast_weight_norm, format_forecast
from flashcart.training.optimizers import optimizer_from_config
from flashcart.training.schedulers import scheduler_from_config
from flashcart.training.tasks import EvaluationTask, TrainingTask, unwrap_model
from flashcart.utils.config import load_config, parse_config_args, save_yaml
from flashcart.utils.distributed import fabric_from_config, is_global_zero, per_rank_batch_size
from flashcart.utils.logging import setup_logger

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def train(**overrides) -> None:
    """Initialize statistics, train the model, and evaluate its checkpoint.

    Training resumes from the run's saved state when available. After training, the best
    checkpoint is evaluated on the training and validation splits. The final checkpoint
    is used when no best checkpoint was saved.

    Args:
        **overrides: Config keys merged over ``flashcart/configs/default.yaml`` by
            ``load_config`` (model architecture, data paths and keys, losses,
            optimizer/scheduler, distributed and compile settings).
    """
    cfg = load_config(overrides)

    fabric = fabric_from_config(cfg)
    fabric.launch()

    output_path = Path(cfg["output_path"])
    output_path.mkdir(parents=True, exist_ok=True)
    log = setup_logger("flashcart.train", output_path / "train.log", enabled=is_global_zero(fabric))

    model_seed = int(cfg.get("model_seed", 0))
    data_seed = int(cfg.get("data_seed", 0))
    cfg["model_seed"] = model_seed
    cfg["data_seed"] = data_seed
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(model_seed)

    if cfg.get("elements") is None:
        cfg["elements"] = get_elements(cfg["train_path"], cfg["energy_key"], cfg["forces_key"])
    if is_global_zero(fabric):
        save_yaml(output_path / "config.yaml", cfg)
        log.info(f"Elements ({len(cfg['elements'])}): {cfg['elements']}")
        log.info("Seeds: model=%s data=%s", model_seed, data_seed)
    if fabric.world_size > 1:
        fabric.barrier()

    world_size = fabric.world_size
    rank = fabric.global_rank
    train_batch_size = per_rank_batch_size(cfg.get("train_batch_size", 64), world_size)
    eval_batch_size = per_rank_batch_size(cfg.get("eval_batch_size", 64), world_size)
    if world_size > 1 and is_global_zero(fabric):
        log.info(
            "Using global batch sizes train=%s eval=%s across %s ranks. Per-rank train=%s eval=%s.",
            cfg.get("train_batch_size", 64),
            cfg.get("eval_batch_size", 64),
            world_size,
            train_batch_size,
            eval_batch_size,
        )

    model = FlashCartPotential.from_config(cfg)

    if is_global_zero(fabric):
        prefit_loader = make_loader(
            cfg["train_path"],
            cfg["elements"],
            model.r_max,
            batch_size=cfg.get("train_batch_size", 64),
            energy_key=cfg["energy_key"],
            forces_key=cfg["forces_key"],
            shuffle=False,
            dynamic_batching=cfg.get("dynamic_batching", True),
            data_seed=data_seed,
        )
        model.pre_fit(prefit_loader)
        del prefit_loader
        if world_size > 1:
            log.info("Pre-fit statistics computed on rank 0.")
    if world_size > 1:
        fabric.barrier()
        model.broadcast_pre_fit_state(fabric)

    resume_state_exists = (output_path / "log" / "training_state.pt").exists()
    if cfg.get("checkpoint") and not resume_state_exists:
        checkpoint = Path(cfg["checkpoint"])
        model.load_weights(checkpoint)
        if is_global_zero(fabric):
            log.info("Initialized model weights from checkpoint: %s", checkpoint)
    elif cfg.get("checkpoint") and resume_state_exists and is_global_zero(fabric):
        log.info("Ignoring checkpoint=%s because %s will be auto-resumed.", cfg["checkpoint"], output_path / "log")

    atomic_shifts = model.scale_shift.shifts.detach().cpu().tolist()

    n_workers = int(cfg.get("n_workers", 0))
    train_loader = make_loader(
        cfg["train_path"],
        cfg["elements"],
        model.r_max,
        batch_size=train_batch_size,
        energy_key=cfg["energy_key"],
        forces_key=cfg["forces_key"],
        shuffle=True,
        dynamic_batching=cfg.get("dynamic_batching", True),
        n_replicas=world_size,
        rank=rank,
        drop_last=cfg.get("drop_last", False),
        balance_batches=world_size > 1,
        data_seed=data_seed,
        atomic_shifts=atomic_shifts,
        n_workers=n_workers,
    )
    valid_loader = make_loader(
        cfg["valid_path"],
        cfg["elements"],
        model.r_max,
        batch_size=eval_batch_size,
        energy_key=cfg["energy_key"],
        forces_key=cfg["forces_key"],
        shuffle=False,
        dynamic_batching=cfg.get("dynamic_batching", True),
        n_replicas=world_size,
        rank=rank,
        data_seed=data_seed,
        atomic_shifts=atomic_shifts,
        n_workers=n_workers,
    )

    if is_global_zero(fabric):
        log.info(str(model))
        log.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_loss = loss_from_config(cfg["train_loss"])
    early_stopping_loss = loss_from_config(cfg["early_stopping_loss"])
    eval_losses = [loss_from_config(s) for s in cfg["valid_losses"]]

    optimizer, aux_optimizers = optimizer_from_config(model, cfg)
    scheduler = scheduler_from_config(optimizer, cfg)

    if aux_optimizers and is_global_zero(fabric):
        # Estimate how weight decay and the learning-rate schedule affect weight norms.
        forecast = forecast_weight_norm(
            epoch_learning_rates(cfg),
            len(train_loader),
            float(cfg.get("weight_decay", 0.0)),
        )
        log.info(format_forecast(forecast))

    task = TrainingTask(
        model=model,
        train_loss=train_loss,
        early_stopping_loss=early_stopping_loss,
        eval_losses=eval_losses,
        optimizer=optimizer,
        scheduler=scheduler,
        aux_optimizers=aux_optimizers,
        max_epochs=cfg.get("max_epochs", 100),
        validate_every=cfg.get("validate_every", 1),
        save_every=cfg.get("save_every", 50),
        grad_clip_norm=cfg.get("grad_clip_norm", 100.0),
        log_weight_norms=cfg.get("log_weight_norms", False),
        ema_decay=cfg.get("ema_decay"),
        predict_compile=cfg.get("predict_compile", False),
        predict_compile_mode=cfg.get("predict_compile_mode", "default"),
        predict_compile_fullgraph=cfg.get("predict_compile_fullgraph", True),
        predict_compile_dynamic=cfg.get("predict_compile_dynamic", False),
        predict_pad_atom_multiple=cfg.get("predict_pad_atom_multiple", 128),
        predict_pad_edge_multiple=cfg.get("predict_pad_edge_multiple", 128),
        predict_pad_graph_multiple=cfg.get("predict_pad_graph_multiple", 8),
    )

    task.run(train_loader, valid_loader, output_path=output_path, fabric=fabric)

    if is_global_zero(fabric):
        log.info("Training complete. Evaluating best model on train and validation splits.")
    best_checkpoint = output_path / "best"
    if not (best_checkpoint / "model.pt").exists():
        best_checkpoint = output_path / "log"
        if is_global_zero(fabric):
            log.info("No best checkpoint was written; evaluating final log checkpoint instead.")

    eval_model = unwrap_model(task.model)
    eval_model.load_weights(best_checkpoint)
    eval_task = EvaluationTask(
        eval_model,
        eval_losses,
        predict_compile=cfg.get("predict_compile", False),
        predict_compile_mode=cfg.get("predict_compile_mode", "reduce-overhead"),
        predict_compile_fullgraph=cfg.get("predict_compile_fullgraph", True),
        predict_compile_dynamic=cfg.get("predict_compile_dynamic", False),
        predict_pad_atom_multiple=cfg.get("predict_pad_atom_multiple", 128),
        predict_pad_edge_multiple=cfg.get("predict_pad_edge_multiple", 128),
        predict_pad_graph_multiple=cfg.get("predict_pad_graph_multiple", 8),
    )
    train_eval_loader = make_loader(
        cfg["train_path"],
        cfg["elements"],
        model.r_max,
        batch_size=eval_batch_size,
        energy_key=cfg["energy_key"],
        forces_key=cfg["forces_key"],
        shuffle=False,
        dynamic_batching=cfg.get("dynamic_batching", True),
        n_replicas=world_size,
        rank=rank,
        data_seed=data_seed,
        atomic_shifts=atomic_shifts,
        n_workers=n_workers,
    )
    for label, loader in [("train", train_eval_loader), ("valid", valid_loader)]:
        metrics = eval_task.run(loader, fabric=task.fabric)
        if is_global_zero(fabric):
            log.info(f"[{label}] " + "  ".join(f"{k}: {v:.4g}" for k, v in metrics.items()))


def main() -> None:
    """CLI entry point: ``flashcart-train [CONFIG.yaml] [KEY=VALUE ...]``."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_and_overrides",
        nargs="*",
        metavar="[CONFIG.yaml] [KEY=VALUE ...]",
    )
    args = parser.parse_args()
    train(**parse_config_args(args.config_and_overrides))


if __name__ == "__main__":
    main()
