import logging
from pathlib import Path


def setup_logger(name: str, log_file: Path, enabled: bool = True) -> logging.Logger:
    """Configure a named logger to write to standard error and a file.

    Existing handlers are removed and closed before the new handlers are installed.
    Disabled logging uses a ``NullHandler`` and does not create the output file.

    Args:
        name (str): Logger name.
        log_file (Path): Log file, parents created.
        enabled (bool, optional): False installs only a NullHandler. Default: True.

    Returns:
        logging.Logger: Configured logger with INFO level and propagation disabled.
    """
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    log.propagate = False
    for handler in list(log.handlers):
        log.removeHandler(handler)
        handler.close()

    if not enabled:
        log.addHandler(logging.NullHandler())
        return log

    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(), logging.FileHandler(log_file)):
        handler.setFormatter(fmt)
        log.addHandler(handler)
    return log
