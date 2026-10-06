import math
from typing import Any, Dict, Optional

import torch

_SHAPES = ("linear", "cosine", "sqrt")


# Both fraction helpers map progress in [0, 1] to a factor in [0, 1]:
# warmup rises 0 -> 1, decay falls 1 -> 0, with the requested curvature.
def _warmup_fraction(progress: float, shape: str) -> float:
    if shape == "linear":
        return progress
    if shape == "cosine":
        return 0.5 * (1.0 - math.cos(math.pi * progress))
    return math.sqrt(progress)


def _decay_fraction(progress: float, shape: str) -> float:
    if shape == "linear":
        return 1.0 - progress
    if shape == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return 1.0 - math.sqrt(progress)


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    start_factor: float = 0.1,
    end_factor: float = 1e-3,
    warmup_epochs: int = 0,
    max_epochs: int = 100,
    decay_epochs: int = 0,
    warmup_shape: str = "linear",
    decay_shape: str = "cosine",
    **kwargs: Any,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Construct an epoch-based warmup, constant, and decay schedule.

    The learning-rate factor rises from ``start_factor`` to one during warmup.
    It remains constant until the final ``decay_epochs``, when it decreases to
    ``end_factor``. A zero warmup or decay length disables that phase.

    When decay is enabled, ``end_factor`` is reached at scheduler epoch
    ``max_epochs``. If the scheduler is stepped after each training epoch,
    this occurs after the final optimizer step.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer to schedule.
        start_factor (float, optional): Initial LR fraction, in (0, 1]. Default: 0.1.
        end_factor (float, optional): Final LR fraction, in [0, 1]. Default: 1e-3.
        warmup_epochs (int, optional): Warmup length. Default: 0.
        max_epochs (int, optional): Total epochs. Default: 100.
        decay_epochs (int, optional): Decay length (counted from the end). Default: 0.
        warmup_shape (str, optional): "linear", "cosine", or "sqrt". Default: "linear".
        decay_shape (str, optional): "linear", "cosine", or "sqrt". Default: "cosine".
        **kwargs: Additional keyword arguments, accepted but unused.

    Returns:
        torch.optim.lr_scheduler.LambdaLR: Epoch-based learning-rate schedule.
    """
    if not (0.0 < start_factor <= 1.0):
        raise ValueError(f"Start factor must be in (0, 1]. Provided: {start_factor=}.")
    if not (0.0 <= end_factor <= 1.0):
        raise ValueError(f"End factor must be in [0, 1]. Provided: {end_factor=}.")
    for name, shape in (("warmup_shape", warmup_shape), ("decay_shape", decay_shape)):
        if shape not in _SHAPES:
            raise ValueError(f"Unsupported {name}={shape!r}. Expected one of {_SHAPES}.")

    n_warmup = max(int(warmup_epochs), 0)
    n_decay = max(int(decay_epochs), 0)
    n_stable = int(max_epochs) - n_warmup - n_decay
    if n_stable < 0:
        raise ValueError(
            f"Warmup epochs ({n_warmup=}) + decay epochs ({n_decay=}) exceed maximal number of epochs ({max_epochs=})."
        )

    def lr_lambda(epoch: int) -> float:
        if n_warmup > 0 and epoch < n_warmup:
            return start_factor + (1.0 - start_factor) * _warmup_fraction(epoch / n_warmup, warmup_shape)
        if n_decay == 0 or epoch < n_warmup + n_stable:
            return 1.0
        progress = min(max((epoch - n_warmup - n_stable) / n_decay, 0.0), 1.0)
        return end_factor + (1.0 - end_factor) * _decay_fraction(progress, decay_shape)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def scheduler_from_config(
    optimizer: torch.optim.Optimizer, cfg: Dict[str, Any]
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """Construct a learning-rate schedule from a run configuration.

    This factory reads the supplied mapping without loading packaged defaults.
    Omitted warmup and decay lengths are zero.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer to schedule.
        cfg (dict): Run config (keys as in ``make_scheduler``).

    Returns:
        LRScheduler | None: Configured schedule, or None when both warmup and decay have
            zero length.
    """
    warmup_epochs = max(int(cfg.get("warmup_epochs", 0)), 0)
    decay_epochs = max(int(cfg.get("decay_epochs", 0)), 0)
    if warmup_epochs == 0 and decay_epochs == 0:
        return None

    return make_scheduler(
        optimizer,
        start_factor=cfg.get("start_factor", 0.1),
        end_factor=cfg.get("end_factor", 1e-3),
        warmup_epochs=warmup_epochs,
        max_epochs=int(cfg.get("max_epochs", 100)),
        decay_epochs=decay_epochs,
        warmup_shape=cfg.get("warmup_shape", "linear"),
        decay_shape=cfg.get("decay_shape", "cosine"),
    )
