try:
    import torch._dynamo as dynamo
except ImportError:
    dynamo = None

from torch import autograd

_RECOMPILE_LIMIT = 8


def configure_autograd_for_compile(allow_autograd: bool = True) -> None:
    """Configure Dynamo to trace prediction functions that differentiate energy.

    The function controls whether ``torch.autograd.grad`` is permitted in compiled
    graphs, raises the recompilation limit to at least eight, and disables Dynamo's DDP
    optimization when that option is available. These settings affect the process
    globally. If Dynamo is unavailable, the function returns without making changes.

    Args:
        allow_autograd (bool, optional): Allow (True) or forbid (False) autograd.grad
            inside compiled graphs. Default: True.
    """
    if dynamo is None:
        return

    if getattr(dynamo.config, "recompile_limit", 0) < _RECOMPILE_LIMIT:
        dynamo.config.recompile_limit = _RECOMPILE_LIMIT

    if hasattr(dynamo.config, "optimize_ddp"):
        dynamo.config.optimize_ddp = False

    if allow_autograd:
        dynamo.allow_in_graph(autograd.grad)
        if hasattr(dynamo.config, "trace_autograd_ops"):
            dynamo.config.trace_autograd_ops = True
    else:
        dynamo.disallow_in_graph(autograd.grad)
