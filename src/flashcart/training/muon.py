"""Hybrid Muon optimizer with batched Newton-Schulz orthogonalisation.

Parts of this module (the Newton-Schulz iteration and the learning-rate
adjustment) are adapted from ``torch.optim._muon`` in PyTorch,
Copyright (c) 2016- Facebook, Inc. and other contributors, BSD 3-Clause
License (see NOTICE).
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import Tensor
from torch.optim._muon import Muon
from torch.optim.optimizer import _to_scalar

NS_COEFF_FAST: Tuple[float, float, float] = (3.4445, -4.7750, 2.0315)
NS_COEFF_POLISH: Tuple[float, float, float] = (2.0, -1.5, 0.5)

NS_STEPS = 5
NS_POLISH_STEPS = 0

NS_RMS_SCALE = 0.2


def _batched_newtonschulz(
    update: Tensor,
    ns_coefficients: Tuple[float, float, float],
    ns_steps: int,
    eps: float,
    polish_coefficients: Tuple[float, float, float] = NS_COEFF_POLISH,
    polish_steps: int = 0,
) -> Tensor:
    """Apply Newton-Schulz iterations to a batch of matrix updates.

    Args:
        update (Tensor): Matrices with shape ``(batch_size, rows, columns)``.
        ns_coefficients (tuple[float, float, float]): Main iteration coefficients.
        ns_steps (int): Number of main iterations.
        eps (float): Lower bound on each matrix's normalization denominator.
        polish_coefficients (tuple[float, float, float], optional): Polishing
            coefficients. Default: ``NS_COEFF_POLISH``.
        polish_steps (int, optional): Number of polishing iterations. Default: 0.

    Returns:
        Tensor: Transformed matrices with the input shape and dtype ``bfloat16``.
    """
    x = update.bfloat16()
    transposed = x.size(1) > x.size(2)
    if transposed:
        x = x.transpose(1, 2)
    x = x / x.norm(dim=(1, 2), keepdim=True).clamp(min=eps)
    for coefficients, steps in ((ns_coefficients, ns_steps), (polish_coefficients, polish_steps)):
        a, b, c = coefficients
        for _ in range(steps):
            gram = torch.bmm(x, x.transpose(1, 2))
            gram_update = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
            x = torch.baddbmm(x, gram_update, x, beta=a)
    return x.transpose(1, 2) if transposed else x


def _adjust_lr(lr: float, adjust_lr_fn: Optional[str], param_shape: torch.Size, rms_scale: float) -> float:
    """``torch.optim._muon._adjust_lr`` with the match-RMS constant exposed.

    Args:
        lr (float): Base learning rate.
        adjust_lr_fn (str | None): "original", "match_rms_adamw", or None.
        param_shape (torch.Size): Shape of the parameter being updated.
        rms_scale (float): The match-RMS constant (``NS_RMS_SCALE``).

    Returns:
        float: Learning rate adjusted for the parameter shape and rescaling rule.
    """
    a, b = param_shape[:2]
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        return lr * math.sqrt(max(1, a / b))
    if adjust_lr_fn == "match_rms_adamw":
        return lr * rms_scale * math.sqrt(max(a, b))
    return lr


@dataclass
class _Bucket:
    shape: Tuple[int, ...]
    params: List[Tensor] = field(default_factory=list)
    names: List[str] = field(default_factory=list)
    bufs: List[Tensor] = field(default_factory=list)
    scores: List[Tensor] = field(default_factory=list)
    damping_tau: float = 2.0
    damping_ema_decay: float = 0.9
    damping_min_scale: float = 0.1
    sigmoid_min: float = 0.0
    sigmoid_max: float = 1.0
    nonfinite: Optional[Tensor] = None


class HybridMuon(Muon):
    """Batched Muon with optional Magma alignment damping.

    Follows the Muon optimizer used to train DPA4 (arXiv:2606.02419).

    Parameters with the same shape, device, and dtype are grouped for batched
    Newton-Schulz updates. Optional alignment damping rescales each update using a
    moving average of the alignment between the gradient and momentum.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining
            parameter groups.
        alignment_damping (bool, optional): Whether to apply alignment damping. Default:
            False.
        damping_tau (float, optional): Tau parameter for the alignment damping sigmoid.
            Default: 2.0.
        damping_ema_decay (float, optional): EMA decay for the alignment damping score.
            Default: 0.9.
        damping_min_scale (float, optional): Minimum scale for the alignment damping.
            Default: 0.1.
        ns_polish_steps (int, optional): Exact-Newton polish iterations after the fast
            ones. Default: 0 (torch behaviour).
        ns_polish_coefficients (tuple, optional): Coefficients for those iterations.
            Default: (2.0, -1.5, 0.5).
        ns_rms_scale (float, optional): Match-RMS rescale constant. Default: 0.2.
        **muon_kwargs: Additional keyword arguments for the base Muon optimizer.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        alignment_damping: bool = False,
        damping_tau: float = 2.0,
        damping_ema_decay: float = 0.9,
        damping_min_scale: float = 0.1,
        ns_polish_steps: int = 0,
        ns_polish_coefficients: Tuple[float, float, float] = NS_COEFF_POLISH,
        ns_rms_scale: float = NS_RMS_SCALE,
        **muon_kwargs: Any,
    ) -> None:
        super().__init__(params, **muon_kwargs)
        extra = {
            "alignment_damping": alignment_damping,
            "damping_tau": damping_tau,
            "damping_ema_decay": damping_ema_decay,
            "damping_min_scale": damping_min_scale,
            "ns_polish_steps": ns_polish_steps,
            "ns_polish_coefficients": ns_polish_coefficients,
            "ns_rms_scale": ns_rms_scale,
        }
        self.defaults.update(extra)
        for group in self.param_groups:
            for key, value in extra.items():
                group.setdefault(key, value)
        self._routing: Any = None
        self._foreach_lists: Any = None
        self._routing_sig: Any = None

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore the optimizer state and reset the internal parameter grouping.

        Missing parameter-group settings are filled from the optimizer defaults.
        Parameters are matched by their order within each group, not by name.

        Args:
            state_dict (dict): Optimizer state returned by ``state_dict()``.
        """
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            for key, value in self.defaults.items():
                group.setdefault(key, value)
        self._routing = None  # load replaces state tensor objects

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        super().add_param_group(param_group)
        self._routing = None

    def _build_routing(self, sig: Tuple[int, ...]) -> None:
        routing, foreach_lists = [], []
        for group in self.param_groups:
            damping = group["alignment_damping"]
            tau = group["damping_tau"]
            names = group.get("param_names")
            buckets: Dict[Any, _Bucket] = {}
            flats: Dict[Any, list] = {}
            for index, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad, memory_format=torch.preserve_format)
                if damping and "magma_score" not in state:
                    state["magma_score"] = torch.full((), 0.5, dtype=torch.float32, device=p.device)
                key = (tuple(p.shape), p.device, p.dtype)
                bucket = buckets.get(key)
                if bucket is None:
                    bucket = buckets[key] = _Bucket(
                        shape=tuple(p.shape),
                        damping_tau=tau,
                        damping_ema_decay=group["damping_ema_decay"],
                        damping_min_scale=group["damping_min_scale"],
                        # Stretch sigmoid(cos / tau) so cos=-1 -> 0 and cos=+1 -> 1.
                        sigmoid_min=1.0 / (1.0 + math.exp(1.0 / tau)),
                        sigmoid_max=1.0 / (1.0 + math.exp(-1.0 / tau)),
                    )
                bucket.params.append(p)
                bucket.names.append(names[index] if names else f"param[{index}]{tuple(p.shape)}")
                bucket.bufs.append(state["momentum_buffer"])
                if damping:
                    bucket.scores.append(state["magma_score"])
                flats.setdefault((p.device, p.dtype), []).append(p)
            for bucket in buckets.values():
                bucket.nonfinite = torch.zeros(len(bucket.params), dtype=torch.bool, device=bucket.params[0].device)
            routing.append(buckets)
            foreach_lists.append(list(flats.values()))
        self._routing = routing
        self._foreach_lists = foreach_lists
        self._routing_sig = sig

    def nonfinite_report(self) -> Dict[str, bool]:
        """Report and clear recorded non-finite gradient flags.

        Returns:
            dict[str, bool]: Parameter names mapped to True when a non-finite gradient
                was recorded in the current parameter routing.
        """
        report: Dict[str, bool] = {}
        if self._routing is None:
            return report
        for buckets in self._routing:
            for bucket in buckets.values():
                if bucket.nonfinite is None:
                    continue
                flags = bucket.nonfinite.tolist()  # single sync per bucket
                if any(flags):
                    for name, flag in zip(bucket.names, flags):
                        if flag:
                            report[name] = True
                    bucket.nonfinite.zero_()
        return report

    def _batched_magma_scale(self, grads_stack: Tensor, bufs_stack: Tensor, bucket: _Bucket) -> Tensor:
        g = grads_stack.reshape(grads_stack.size(0), -1)
        m = bufs_stack.reshape(bufs_stack.size(0), -1)
        if g.dtype != torch.float32:
            g = g.float()
        if m.dtype != torch.float32:
            m = m.float()
        cosine = ((m * g).sum(dim=1) / (m.norm(dim=1) * g.norm(dim=1)).clamp(min=1e-12)).clamp(-1.0, 1.0)
        raw_score = (
            (torch.sigmoid(cosine / bucket.damping_tau) - bucket.sigmoid_min)
            / (bucket.sigmoid_max - bucket.sigmoid_min)
        ).clamp(min=0.0, max=1.0)
        scores = torch.stack(bucket.scores)
        scores.mul_(bucket.damping_ema_decay).add_(raw_score, alpha=1.0 - bucket.damping_ema_decay)
        torch._foreach_copy_(bucket.scores, list(scores.unbind(0)))
        return scores * (1.0 - bucket.damping_min_scale) + bucket.damping_min_scale

    @torch.no_grad()
    def step(self, closure=None):
        """Update parameters using the current gradients.

        Args:
            closure (callable, optional): Function that reevaluates the loss with
                gradient recording enabled. Default: None.

        Returns:
            torch.Tensor | None: Loss returned by the closure, or None when no closure
                is provided.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        sig = tuple(id(p) for g in self.param_groups for p in g["params"] if p.grad is not None)
        if self._routing is None or sig != self._routing_sig:
            self._build_routing(sig)

        for group, buckets, flats in zip(self.param_groups, self._routing, self._foreach_lists):
            if not buckets:
                continue
            lr = _to_scalar(group["lr"])
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            damping = group["alignment_damping"]

            if weight_decay != 0.0:
                for flat_params in flats:
                    torch._foreach_mul_(flat_params, 1 - lr * weight_decay)

            for (shape, device, dtype), bucket in buckets.items():
                grads_stack = torch.stack([p.grad for p in bucket.params])
                bucket.nonfinite.logical_or_(~torch.isfinite(grads_stack).flatten(1).all(dim=1))
                bufs_stack = torch.stack(bucket.bufs)
                scale = self._batched_magma_scale(grads_stack, bufs_stack, bucket) if damping else None
                bufs_stack.lerp_(grads_stack, 1 - momentum)
                torch._foreach_copy_(bucket.bufs, list(bufs_stack.unbind(0)))
                update = torch.lerp(grads_stack, bufs_stack, momentum) if group["nesterov"] else bufs_stack

                update = _batched_newtonschulz(
                    update,
                    group["ns_coefficients"],
                    group["ns_steps"],
                    group["eps"],
                    group["ns_polish_coefficients"],
                    group["ns_polish_steps"],
                )

                if scale is not None:
                    update = (update * scale.view(-1, 1, 1)).to(dtype)
                else:
                    update = update.to(dtype)
                adjusted_lr = _adjust_lr(lr, group["adjust_lr_fn"], bucket.params[0].shape, group["ns_rms_scale"])
                torch._foreach_add_(bucket.params, list(update.unbind(0)), alpha=-adjusted_lr)
        return loss
