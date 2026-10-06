from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def make_optimizer(
    model: nn.Module,
    optimizer: str = "muon",
    lr: float = 1e-2,
    weight_decay: float = 1e-3,
    muon_alignment_damping: bool = False,
    muon_damping_tau: float = 2.0,
    muon_damping_ema_decay: float = 0.9,
    muon_damping_min_scale: float = 0.1,
    muon_ns_steps: Optional[int] = None,
    muon_ns_polish_steps: Optional[int] = None,
    muon_ns_rms_scale: Optional[float] = None,
) -> Tuple[torch.optim.Optimizer, Optional[List[torch.optim.Optimizer]]]:
    """Construct optimizers from the model's parameter-selection hooks.

    With ``optimizer="muon"``, trainable matrix parameters not excluded by
    ``non_muon_parameters`` are assigned to ``HybridMuon``. Remaining parameters are
    assigned to AdamW, with weight decay disabled for those returned by
    ``non_decayable_parameters``. With ``optimizer="adamw"``, AdamW handles all
    parameter groups.

    The returned AdamW optimizer is the main optimizer. ``TrainingTask`` applies its
    learning-rate schedule to any auxiliary Muon optimizer.

    Args:
        model (nn.Module): Model providing the parameter-group hooks.
        optimizer (str, optional): "muon" or "adamw". Default: "muon".
        lr (float, optional): Learning rate for all groups. Default: 1e-2.
        weight_decay (float, optional): Decay for the decayable groups. Default: 1e-3.
        muon_alignment_damping (bool, optional): Magma alignment damping in HybridMuon.
            Default: False.
        muon_damping_tau (float, optional): Damping sigmoid temperature. Default: 2.0.
        muon_damping_ema_decay (float, optional): Damping score EMA decay. Default: 0.9.
        muon_damping_min_scale (float, optional): Damping floor. Default: 0.1.
        muon_ns_steps (int, optional): Number of Newton-Schulz iterations. Default:
            None, using ``NS_STEPS`` (5).
        muon_ns_polish_steps (int, optional): Number of polishing iterations after the
            Newton-Schulz steps.
            Default: None, using ``NS_POLISH_STEPS`` (0).
        muon_ns_rms_scale (float, optional): Rescaling factor for the normalized update.
            Default: None, using ``NS_RMS_SCALE`` (0.2).

    Returns:
        tuple: Main AdamW optimizer and a list of auxiliary Muon optimizers, or None for
            the auxiliary entry when ``optimizer="adamw"``.
    """
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]

    non_muon: set = set()
    non_decayable: set = set()
    if hasattr(model, "non_muon_parameters"):
        non_muon = set(model.non_muon_parameters())
    if hasattr(model, "non_decayable_parameters"):
        non_decayable = set(model.non_decayable_parameters())

    muon_named = [(name, p) for name, p in named if p not in non_muon and p.ndim == 2]
    muon_params = [p for _, p in muon_named]

    adam_params = [p for _, p in named if p in non_muon or p.ndim != 2]
    adam_decay = [p for p in adam_params if p not in non_decayable]
    adam_no_decay = [p for p in adam_params if p in non_decayable]

    if optimizer == "muon":
        if weight_decay == 0.0:
            raise ValueError(
                "optimizer='muon' with weight_decay=0.0 leaves the weight norm unbounded: "
                "Newton-Schulz normalizes the update, so it does not shrink as the weights "
                "grow. Size a positive value with forecast_weight_norm in flashcart/training/muon_norm.py."
            )
        aux_optimizers: List[torch.optim.Optimizer] = []
        if muon_named:
            from flashcart.training import muon as muon_module

            aux_optimizers.append(
                muon_module.HybridMuon(
                    muon_named,
                    alignment_damping=muon_alignment_damping,
                    damping_tau=muon_damping_tau,
                    damping_ema_decay=muon_damping_ema_decay,
                    damping_min_scale=muon_damping_min_scale,
                    ns_steps=muon_module.NS_STEPS if muon_ns_steps is None else muon_ns_steps,
                    ns_polish_steps=(
                        muon_module.NS_POLISH_STEPS if muon_ns_polish_steps is None else muon_ns_polish_steps
                    ),
                    ns_rms_scale=muon_module.NS_RMS_SCALE if muon_ns_rms_scale is None else muon_ns_rms_scale,
                    lr=lr,
                    adjust_lr_fn="match_rms_adamw",
                    weight_decay=weight_decay,
                )
            )
        main_optimizer = torch.optim.AdamW(
            [
                {"params": adam_decay, "weight_decay": weight_decay},
                {"params": adam_no_decay, "weight_decay": 0.0},
            ],
            lr=lr,
        )
        return main_optimizer, aux_optimizers

    if optimizer == "adamw":
        main_optimizer = torch.optim.AdamW(
            [
                {"params": muon_params, "weight_decay": weight_decay},
                {"params": adam_decay, "weight_decay": weight_decay},
                {"params": adam_no_decay, "weight_decay": 0.0},
            ],
            lr=lr,
        )
        return main_optimizer, None

    raise ValueError(f"Unsupported optimizer={optimizer!r}. Expected 'muon' or 'adamw'.")


def optimizer_from_config(
    model: nn.Module, cfg: Dict[str, Any]
) -> Tuple[torch.optim.Optimizer, Optional[List[torch.optim.Optimizer]]]:
    """Construct optimizers from a run configuration.

    This factory reads the supplied mapping without loading packaged defaults.
    Omitted settings use the defaults of ``make_optimizer``.

    Args:
        model (nn.Module): Model to optimize.
        cfg (dict): Run config (keys as in ``make_optimizer``).

    Returns:
        tuple: Main AdamW optimizer and a list of auxiliary Muon optimizers, or None for
            the auxiliary entry when ``optimizer="adamw"``.
    """
    return make_optimizer(
        model,
        optimizer=cfg.get("optimizer", "muon"),
        lr=cfg.get("lr", 1e-2),
        weight_decay=cfg.get("weight_decay", 1e-3),
        muon_alignment_damping=cfg.get("muon_alignment_damping", False),
        muon_damping_tau=cfg.get("muon_damping_tau", 2.0),
        muon_damping_ema_decay=cfg.get("muon_damping_ema_decay", 0.9),
        muon_damping_min_scale=cfg.get("muon_damping_min_scale", 0.1),
        muon_ns_steps=cfg.get("muon_ns_steps"),
        muon_ns_polish_steps=cfg.get("muon_ns_polish_steps"),
        muon_ns_rms_scale=cfg.get("muon_ns_rms_scale"),
    )
