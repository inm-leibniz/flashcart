"""Evaluate a saved FlashCart model (the ``flashcart-test`` console script)."""

import argparse
import os
from pathlib import Path

import torch

from flashcart.data.dataset import make_loader
from flashcart.data.utils import get_elements
from flashcart.model.flashcart import FlashCartPotential
from flashcart.training.loss import loss_from_config
from flashcart.training.tasks import EvaluationTask, unwrap_model
from flashcart.utils.config import load_config, load_yaml, parse_config_args
from flashcart.utils.distributed import fabric_from_config, is_global_zero, per_rank_batch_size
from flashcart.utils.logging import setup_logger

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def test(**overrides) -> dict:
    """Evaluate a saved checkpoint on the configured validation and test datasets.

    The model architecture comes from the checkpoint's ``model_config.yaml``.
    Evaluation settings come from the packaged defaults and supplied overrides.
    The checkpoint is selected by ``checkpoint`` or defaults to ``output_path/best``.
    For older checkpoints without a model configuration, the model is reconstructed
    from the run's ``config.yaml`` and applicable overrides.

    Args:
        **overrides: Configuration settings merged by ``load_config``, including
            ``checkpoint`` or ``output_path``, ``valid_path`` and ``test_path``,
            ``valid_losses``, reference-property keys, and device and compilation
            options.

    Returns:
        dict: Mapping from each evaluated split name to its metric names and values.

    Raises:
        ValueError: No checkpoint location or evaluation dataset is specified, or an
            older checkpoint lacks the information needed to determine its elements.
    """
    cfg = load_config(overrides, required=[])
    checkpoint = Path(cfg["checkpoint"]) if cfg.get("checkpoint") else None
    if checkpoint is None:
        output_path = cfg.get("output_path")
        if output_path is None:
            raise ValueError("Provide 'checkpoint' or 'output_path'.")
        checkpoint = Path(output_path) / "best"

    fabric = fabric_from_config(cfg)
    fabric.launch()
    log = setup_logger("flashcart.test", checkpoint.parent / "test.log", enabled=is_global_zero(fabric))
    model_seed = int(cfg.get("model_seed", 0))
    data_seed = int(cfg.get("data_seed", 0))
    torch.manual_seed(model_seed)

    valid_losses = [loss_from_config(s) for s in cfg["valid_losses"]]

    if (checkpoint / "model_config.yaml").exists() or not (checkpoint.parent / "config.yaml").exists():
        model = FlashCartPotential.from_checkpoint(checkpoint, map_location="cpu")
    else:
        run_cfg = load_yaml(checkpoint.parent / "config.yaml")
        run_cfg.update({k: v for k, v in overrides.items() if k not in ("checkpoint", "output_path")})
        if run_cfg.get("elements") is None:
            train_path = run_cfg.get("train_path")
            if train_path is None:
                raise ValueError("Legacy checkpoints without model_config.yaml need 'elements' or 'train_path'.")
            run_cfg["elements"] = get_elements(
                train_path,
                run_cfg.get("energy_key", "REF_energy"),
                run_cfg.get("forces_key", "REF_forces"),
            )
        model = FlashCartPotential.from_config(run_cfg)
        model.load_weights(checkpoint, map_location="cpu")

    model = fabric.setup(model)
    if hasattr(model, "mark_forward_method"):
        model.mark_forward_method("predict")
    base_model = unwrap_model(model)
    elements = base_model.elements
    eval_batch_size = per_rank_batch_size(cfg.get("eval_batch_size", 64), fabric.world_size)
    atomic_shifts = base_model.scale_shift.shifts.detach().cpu().tolist()

    eval_task = EvaluationTask(
        model,
        valid_losses,
        predict_compile=cfg.get("predict_compile", False),
        predict_compile_mode=cfg.get("predict_compile_mode", "default"),
        predict_compile_fullgraph=cfg.get("predict_compile_fullgraph", True),
        predict_compile_dynamic=cfg.get("predict_compile_dynamic", False),
        predict_pad_atom_multiple=cfg.get("predict_pad_atom_multiple", 128),
        predict_pad_edge_multiple=cfg.get("predict_pad_edge_multiple", 128),
        predict_pad_graph_multiple=cfg.get("predict_pad_graph_multiple", 8),
    )

    splits = {}
    if cfg.get("valid_path"):
        splits["valid"] = make_loader(
            cfg["valid_path"],
            elements,
            base_model.r_max,
            batch_size=eval_batch_size,
            energy_key=cfg["energy_key"],
            forces_key=cfg["forces_key"],
            shuffle=False,
            dynamic_batching=cfg.get("dynamic_batching", True),
            n_replicas=fabric.world_size,
            rank=fabric.global_rank,
            data_seed=data_seed,
            atomic_shifts=atomic_shifts,
        )
    if cfg.get("test_path"):
        splits["test"] = make_loader(
            cfg["test_path"],
            elements,
            base_model.r_max,
            batch_size=eval_batch_size,
            energy_key=cfg["energy_key"],
            forces_key=cfg["forces_key"],
            shuffle=False,
            dynamic_batching=cfg.get("dynamic_batching", True),
            n_replicas=fabric.world_size,
            rank=fabric.global_rank,
            data_seed=data_seed,
            atomic_shifts=atomic_shifts,
        )

    if not splits:
        raise ValueError("Provide at least one of 'valid_path' or 'test_path' in the config.")

    results = {}
    for label, loader in splits.items():
        metrics = eval_task.run(loader, fabric=fabric)
        results[label] = metrics
        if is_global_zero(fabric):
            log.info(f"[{label}] " + "  ".join(f"{k}: {v:.4g}" for k, v in metrics.items()))
    return results


def main() -> None:
    """CLI entry point: ``flashcart-test [CONFIG.yaml] [KEY=VALUE ...]``."""
    parser = argparse.ArgumentParser(description="Evaluate a saved FlashCart model.")
    parser.add_argument(
        "config_and_overrides",
        nargs="*",
        metavar="[CONFIG.yaml] [KEY=VALUE ...]",
    )
    args = parser.parse_args()
    test(**parse_config_args(args.config_and_overrides))


if __name__ == "__main__":
    main()
