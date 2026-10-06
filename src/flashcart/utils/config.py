import argparse
from pathlib import Path
from typing import Any, Dict, Sequence, Union

import yaml

_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"

REQUIRED_KEYS: list[str] = ["train_path", "valid_path", "output_path"]


def save_yaml(path: Union[str, Path], data: Dict[str, Any]) -> None:
    """Write a dict as YAML, creating parent directories.

    Args:
        path (str | Path): Output file.
        data (dict): Mapping to dump (key order preserved).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Read a YAML file.

    Empty files and files containing only ``null`` produce an empty dictionary.
    Other YAML values are returned as parsed. A top-level mapping is not required.

    Args:
        path (str | Path): Input file.

    Returns:
        Any: Parsed YAML value, or an empty dictionary if the file is empty or
            contains only ``null``.
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    return {} if data is None else data


def load_config(
    overrides: Union[str, Path, Dict[str, Any], argparse.Namespace, None] = None,
    required: list[str] = REQUIRED_KEYS,
) -> Dict[str, Any]:
    """Load the packaged defaults and replace settings with the supplied overrides.

    Merging replaces values at the top level. Nested mappings are replaced rather
    than recursively merged. Defaults come from ``flashcart/configs/default.yaml``.

    Args:
        overrides (str | Path | dict | argparse.Namespace, optional): YAML file,
            mapping, or namespace containing overrides. None-valued entries are skipped
            only for namespaces. Default: None.
        required (list[str], optional): Keys that must be non-None in the result. Pass
            [] to skip the check. Default: REQUIRED_KEYS.

    Returns:
        dict[str, Any]: Default configuration updated with the supplied overrides and
            checked for required values.

    Raises:
        ValueError: A required setting is missing or None.
    """
    cfg: Dict[str, Any] = load_yaml(_DEFAULT_CONFIG)

    if overrides is not None:
        if isinstance(overrides, argparse.Namespace):
            user = {k: v for k, v in vars(overrides).items() if v is not None}
        elif isinstance(overrides, dict):
            user = overrides
        else:
            user = load_yaml(overrides)
        cfg.update(user)

    missing = [k for k in required if cfg.get(k) is None]
    if missing:
        raise ValueError(f"Config is missing required keys: {missing}.")
    return cfg


def parse_override(s: str) -> tuple[str, Any]:
    """Parse one ``KEY=VALUE`` command-line override into a key and value.

    Values are interpreted first as integers or floats, then as explicit boolean or null
    values. Remaining values are parsed with ``yaml.safe_load``. The original string is
    retained if YAML parsing fails.

    Args:
        s (str): Override of the form "KEY=VALUE".

    Returns:
        tuple[str, Any]: Override key and parsed value.

    Raises:
        ValueError: The override has no key or does not contain an equals sign.
    """
    key, _, raw = s.partition("=")
    if not key or "=" not in s:
        raise ValueError(f"Invalid override {s!r}; expected KEY=VALUE.")
    for cast in (int, float):
        try:
            return key, cast(raw)
        except ValueError:
            pass
    if raw.lower() in ("true", "false"):
        return key, raw.lower() == "true"
    if raw.lower() in ("null", "none", "~"):
        return key, None
    try:
        return key, yaml.safe_load(raw)
    except yaml.YAMLError:
        return key, raw


def parse_config_args(args: Sequence[str]) -> Dict[str, Any]:
    """Parse the CLI convention ``[CONFIG.yaml] [KEY=VALUE ...]`` into overrides.

    An optional leading YAML file is read first. KEY=VALUE pairs then replace the
    corresponding values.

    Args:
        args (Sequence[str]): CLI arguments after the program name.

    Returns:
        dict[str, Any]: Overrides read from the optional file and subsequent
            assignments. Later assignments replace earlier values for the same key.

    Raises:
        ValueError: The loaded configuration is not a mapping, or an assignment
            does not have the form ``KEY=VALUE``.
    """
    args = list(args)
    overrides: Dict[str, Any] = {}

    if args and "=" not in args[0]:
        config_path = Path(args.pop(0))
        user = load_yaml(config_path)
        if not isinstance(user, dict):
            raise ValueError(f"Config file {config_path} must contain a YAML mapping.")
        overrides.update(user)

    overrides.update(dict(parse_override(s) for s in args))
    return overrides


def torch_device_from_config(device: Any) -> Any:
    """Translate the configuration value ``"gpu"`` to the PyTorch value ``"cuda"``.

    Args:
        device (Any): Device spec from the run config.

    Returns:
        Any: ``"cuda"`` for a case-insensitive ``"gpu"`` string. Other input values are
            returned unchanged.
    """
    if isinstance(device, str) and device.lower() == "gpu":
        return "cuda"
    return device
