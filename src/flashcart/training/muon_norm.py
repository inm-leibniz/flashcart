"""Estimate and measure the RMS magnitude of weights updated by Muon.

The forecast models the squared weight RMS, ``u``, by
``u_next = (1 - lr*wd)**2 * u + (A*lr)**2``, starting from ``u=1``. For constant
learning rate and small ``lr*wd``, the stationary RMS is approximately
``A * sqrt(lr / (2*wd))``. The accumulated quantity ``2 * wd * sum(lr)`` estimates the
relaxation of the squared RMS toward this stationary value.

The model is an empirical approximation. Its fitted coefficient does not account
explicitly for gradient correlations, alignment damping, or changes to the Newton-Schulz
update. The reports compare this forecast with observed weight and momentum norms.
"""

import math
import warnings
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import torch

# The constant ``A`` in the recursion above. Fits for individual runs range
# from 0.810 to 0.995 across seven completed runs.
MUON_NORM_CONSTANT = 0.869

# Runs with weight decay near zero have diverged at aggregate rms(W) of
# ~5.4-5.65. Properly decayed runs have trained cleanly to 3.4.
NORM_WARN = 4.0

# Below this value, weight decay is not binding and the norm is a free random walk.
EFOLDINGS_WARN = 0.3


def epoch_learning_rates(cfg: Mapping[str, Any]) -> List[float]:
    """Replay the configured lr schedule, one entry per epoch.

    Args:
        cfg (Mapping): Run config (``lr``, ``max_epochs``, scheduler keys).

    Returns:
        list[float]: Learning rate used at the start of each training epoch.
    """
    from flashcart.training.schedulers import scheduler_from_config

    lr = float(cfg.get("lr", 1e-2))
    n_epochs = int(cfg.get("max_epochs", 100))
    probe = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=lr)
    scheduler = scheduler_from_config(probe, cfg)
    if scheduler is None:
        return [lr] * n_epochs
    lrs = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for _ in range(n_epochs):
            lrs.append(float(probe.param_groups[0]["lr"]))
            scheduler.step()
    return lrs


def forecast_weight_norm(
    epoch_lrs: Iterable[float],
    steps_per_epoch: int,
    weight_decay: float,
    constant: float = MUON_NORM_CONSTANT,
) -> Dict[str, float]:
    """Integrate the norm recursion over a schedule.

    Args:
        epoch_lrs (iterable of float): Learning rate for each epoch.
        steps_per_epoch (int): Optimizer steps per epoch.
        weight_decay (float): Decoupled weight decay.
        constant (float, optional): ``A``. Default: ``MUON_NORM_CONSTANT``.

    Returns:
        dict[str, float]: ``ceiling`` is the approximate stationary RMS at the largest
            learning rate. ``efoldings`` is the accumulated decay measure. ``peak_rms``
            and ``final_rms`` are estimated from the recurrence.
            ``effective_end_factor`` is the final ``lr / rms(W)`` divided by its largest
            value at the sampled epoch endpoints. The ceiling is infinite when weight
            decay is nonpositive.
    """
    epoch_lrs = list(epoch_lrs)
    # rms(W) = 1.0 at initialization by construction
    u, peak, efoldings = 1.0, 1.0, 0.0
    rel_step_peak = 0.0
    for lr in epoch_lrs:
        decay = (1.0 - lr * weight_decay) ** 2
        kick = (constant * lr) ** 2
        if decay == 1.0:
            u = u + kick * steps_per_epoch
        else:
            u = decay**steps_per_epoch * u + kick * (1.0 - decay**steps_per_epoch) / (1.0 - decay)
        peak = max(peak, u)
        efoldings += 2.0 * weight_decay * lr * steps_per_epoch
        rel_step_peak = max(rel_step_peak, lr / math.sqrt(u))
    lr_max = max(epoch_lrs) if epoch_lrs else 0.0
    ceiling = math.inf if weight_decay <= 0.0 else constant * math.sqrt(lr_max / (2.0 * weight_decay))
    rel_step_final = (epoch_lrs[-1] / math.sqrt(u)) if epoch_lrs else 0.0
    return {
        "ceiling": ceiling,
        "efoldings": efoldings,
        "peak_rms": math.sqrt(peak),
        "final_rms": math.sqrt(u),
        "effective_end_factor": rel_step_final / rel_step_peak if rel_step_peak else 0.0,
    }


def format_forecast(forecast: Mapping[str, float]) -> str:
    """One line summarizing a forecast, with any warnings appended.

    Args:
        forecast (Mapping[str, float]): Output of ``forecast_weight_norm``.

    Returns:
        str: One-line forecast with warnings for the configured diagnostic thresholds.
    """
    warnings = []
    if forecast["peak_rms"] > NORM_WARN:
        warnings.append(f"peak rms(W) > {NORM_WARN} (divergence observed at ~5.4-5.65 in low-wd runs)")
    if forecast["efoldings"] < EFOLDINGS_WARN:
        warnings.append(f"only {forecast['efoldings']:.3f} e-foldings: weight decay is not binding")
    tail = ("  ** " + "; ".join(warnings) + " **") if warnings else ""
    return (
        f"Muon weight-norm forecast: ceiling w*={forecast['ceiling']:.3g}  "
        f"e-foldings={forecast['efoldings']:.3f}  "
        f"peak rms(W)={forecast['peak_rms']:.3g}  final={forecast['final_rms']:.3g}  "
        f"effective end_factor={forecast.get('effective_end_factor', 0.0):.2g}{tail}"
    )


def weight_rms_report(model: torch.nn.Module, muon_optimizers: Iterable[torch.optim.Optimizer] = ()) -> Dict[str, Any]:
    """Report aggregate Muon weight RMS and the largest individual matrix RMS.

    The aggregate includes trainable floating-point matrices owned by the provided
    optimizers. The maximum also considers other trainable floating-point matrices
    unless they are excluded from weight decay. The maximum helps identify
    individual matrices that grow while the aggregate RMS changes little.

    Args:
        model (torch.nn.Module): Unwrapped model.
        muon_optimizers (Iterable[torch.optim.Optimizer], optional): Optimizers whose
            parameters define the aggregate. Default: an empty iterable, giving an
            aggregate RMS of zero.

    Returns:
        dict[str, Any]: Aggregate ``weight_rms``, largest ``weight_rms_max``, and its
            parameter name in ``weight_rms_max_tensor``.
    """
    owned = {id(p) for opt in muon_optimizers for group in opt.param_groups for p in group["params"]}
    no_decay = set()
    if hasattr(model, "non_decayable_parameters"):
        no_decay = {id(p) for p in model.non_decayable_parameters()}

    names, tensors, mine = [], [], []
    for name, p in model.named_parameters():
        if p.ndim != 2 or not p.requires_grad or not torch.is_floating_point(p):
            continue
        if id(p) not in owned and id(p) in no_decay:
            continue
        names.append(name)
        tensors.append(p.detach())
        mine.append(id(p) in owned)
    if not tensors:
        return {"weight_rms": 0.0, "weight_rms_max": 0.0, "weight_rms_max_tensor": ""}

    fros = torch.stack(torch._foreach_norm(tensors)).tolist()
    total_sq, total_n = 0.0, 0
    hi: Tuple[float, str] = (-math.inf, "")
    for name, t, fro, owns in zip(names, tensors, fros, mine):
        rms = fro / t.numel() ** 0.5
        if owns:
            total_sq += fro**2
            total_n += t.numel()
        if rms > hi[0]:
            hi = (rms, name)
    return {
        "weight_rms": (total_sq / total_n) ** 0.5 if total_n else 0.0,
        "weight_rms_max": hi[0] if hi[1] else 0.0,
        "weight_rms_max_tensor": hi[1],
    }


# Muon normalizes each momentum buffer by ``||m||.clamp(min=eps)`` before
# Newton-Schulz, so the update is exactly unit-norm while ``||m|| >= eps`` and
# collapses at the clamp.
MOMENTUM_EPS_WARN = 10.0


def momentum_norm_report(optimizers: Iterable[torch.optim.Optimizer]) -> Dict[str, float]:
    """Report the smallest momentum norm relative to the first available epsilon.

    The epsilon is taken from the first optimizer group that defines it. Comparisons
    therefore assume that the provided optimizers use the same Newton-Schulz
    normalization epsilon.

    Args:
        optimizers (iterable): The Muon optimizers (aux_optimizers).

    Returns:
        dict[str, float]: ``momentum_norm_min_over_eps``, or an empty mapping when no
            usable momentum buffers or epsilon are available.
    """
    norms: List[torch.Tensor] = []
    eps: Optional[float] = None
    for opt in optimizers:
        for group in opt.param_groups:
            if eps is None and "eps" in group:
                eps = float(group["eps"])
        buffers = [s["momentum_buffer"] for s in opt.state.values() if torch.is_tensor(s.get("momentum_buffer"))]
        if buffers:
            norms.append(torch.stack(torch._foreach_norm(buffers)).min())
    if not norms or eps is None or eps <= 0.0:
        return {}
    min_norm = float(torch.stack(norms).min())
    return {"momentum_norm_min_over_eps": min_norm / eps}


def muon_weight_decay(optimizers: Iterable[torch.optim.Optimizer]) -> Optional[float]:
    """Read weight decay from the first available optimizer group.

    Args:
        optimizers (Iterable[torch.optim.Optimizer]): Optimizers searched in order.

    Returns:
        float | None: First group's weight decay, defaulting to zero when absent, or
            None when no groups are available.
    """
    for opt in optimizers:
        for group in opt.param_groups:
            return float(group.get("weight_decay", 0.0))
    return None
