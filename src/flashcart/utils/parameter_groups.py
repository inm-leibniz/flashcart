from typing import Iterator

import torch.nn as nn


def iter_child_special_parameters(module: nn.Module, method_name: str) -> Iterator[nn.Parameter]:
    """Yield each parameter once from child modules at all nesting levels.

    Call the named method on each nested child module that has it. Skip the supplied
    module itself. If several child modules return the same parameter object, yield
    it only once.

    Args:
        module (nn.Module): Module containing the child modules to inspect.
        method_name (str): Name of the generator method to call.

    Yields:
        nn.Parameter: Each reported parameter once, in the order found.
    """
    seen = set()
    for child in module.modules():
        if child is module:
            continue
        method = getattr(child, method_name, None)
        if method is None:
            continue
        for param in method():
            if id(param) not in seen:
                seen.add(id(param))
                yield param
