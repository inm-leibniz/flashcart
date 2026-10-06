"""Configure the maximum tensor rank included in generated kernel modules.

Each kernel backend reads ``FLASHCART_KERNEL_L_MAX`` when imported and retains the
result as ``KERNEL_L_MAX``. The value also contributes to the generated-source
fingerprint. Set the environment variable before importing FlashCart. Changing it does
not update imported backends.
"""
import os

_DEFAULT_KERNEL_L_MAX = 3

_ENV = "FLASHCART_KERNEL_L_MAX"


def kernel_l_max() -> int:
    """Read the configured maximum tensor rank for generated kernels.

    Returns:
        int: Value of ``FLASHCART_KERNEL_L_MAX``, or 3 when it is unset.
    """
    v = os.environ.get(_ENV)
    if v is not None:
        return int(v)
    return _DEFAULT_KERNEL_L_MAX
