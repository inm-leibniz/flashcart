import os


def env_bool(name: str, default: bool = False) -> bool:
    """Interpret an environment variable as a boolean.

    The values ``"1"``, ``"true"``, ``"yes"``, and ``"t"`` are true, ignoring case.
    Other values are false.

    Args:
        name (str): Environment variable to read.
        default (bool, optional): Value used when the variable is unset. Default: False.

    Returns:
        bool: Parsed value, or the default when the variable is unset.
    """
    return os.environ.get(name, str(default)).lower() in ("1", "true", "yes", "t")


def env_int(name: str, default: int) -> int:
    """Read an integer environment variable with a fallback value.

    Args:
        name (str): Environment variable to read.
        default (int): Value used when the variable is unset or cannot be parsed as an
            integer.

    Returns:
        int: Parsed value, or the provided default.
    """
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default
