import math
from typing import Any, Mapping

from lightning.fabric import Fabric


def fabric_from_config(cfg: Mapping[str, Any]) -> Fabric:
    """Build an unlaunched Fabric from the run config's device keys.

    Args:
        cfg (Mapping): Configuration providing ``device`` (default ``"gpu"``),
            ``devices`` and ``strategy`` (default ``"auto"``), ``n_nodes`` (default 1),
            and ``precision`` (default None).

    Returns:
        Fabric: Configured Fabric instance, which has not yet been launched.
    """
    return Fabric(
        accelerator=cfg.get("device", "gpu"),
        devices=cfg.get("devices", "auto"),
        strategy=cfg.get("strategy", "auto"),
        num_nodes=cfg.get("n_nodes", 1),
        precision=cfg.get("precision"),
    )


def per_rank_batch_size(global_batch_size: int, world_size: int) -> int:
    """Split a global batch size across ranks by rounding upward.

    Rounding upward can make the combined batch size exceed the requested global size. A
    world size below one is treated as one.

    Args:
        global_batch_size (int): Requested global batch size, before division across
            ranks.
        world_size (int): Number of ranks.

    Returns:
        int: Ceiling of the global batch size divided by the world size, with a minimum
            of one.
    """
    return max(1, math.ceil(int(global_batch_size) / max(int(world_size), 1)))


def is_global_zero(fabric) -> bool:
    """Determine whether execution is on the global rank zero process.

    Args:
        fabric (Fabric or None): Fabric instance, or None for execution without Fabric.

    Returns:
        bool: True when ``fabric`` is None, its ``global_rank`` is zero, or it has no
            ``global_rank`` attribute.
    """
    return fabric is None or getattr(fabric, "global_rank", 0) == 0
