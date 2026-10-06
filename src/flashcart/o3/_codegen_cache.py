"""Cache generated Python and Triton kernel modules outside the package.

Cached filenames include a fingerprint of the selected code generator, the shared
code-generation helpers, and the configured maximum tensor rank. Changes to these inputs
select a different cache file. Modules already loaded in the current process are reused.

The cache location is ``FLASHCART_CACHE_DIR`` when set, otherwise
``$XDG_CACHE_HOME/flashcart`` or ``~/.cache/flashcart``.
"""

import hashlib
import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Callable

_COMMON_CODEGEN_FILE = Path(__file__).with_name("_codegen_common.py")
FINGERPRINT_PREFIX = "# CODEGEN_FINGERPRINT: "


def cache_dir() -> Path:
    """Return the directory used to cache generated modules.

    Returns:
        Path: ``FLASHCART_CACHE_DIR`` when set, otherwise the FlashCart directory
            under ``XDG_CACHE_HOME`` or ``~/.cache``. User-home notation is expanded.
    """
    root = os.environ.get("FLASHCART_CACHE_DIR")
    if root:
        return Path(root).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "flashcart"


def codegen_fingerprint(codegen_file: Path, kernel_l_max: int) -> str:
    """Compute a fingerprint of the code-generation sources and maximum tensor rank.

    Args:
        codegen_file (Path): The ``_codegen_*.py`` module emitting the source.
        kernel_l_max (int): Maximum tensor rank compiled into the generated module.

    Returns:
        str: Fingerprint identifying the source files and configured rank limit.
    """
    h = hashlib.blake2s(digest_size=12)
    h.update(codegen_file.read_bytes())
    h.update(b"\n")
    h.update(_COMMON_CODEGEN_FILE.read_bytes())
    h.update(b"\n")
    h.update(str(kernel_l_max).encode())
    return h.hexdigest()


def load_generated(
    module_name: str,
    codegen_file: Path,
    kernel_l_max: int,
    build_source: Callable[[int], str],
) -> ModuleType:
    """Import a generated module, emitting it into the cache first if needed.

    The module is registered in ``sys.modules`` under ``module_name`` so pickled
    references and repeated imports resolve to the same object. Writes go through a
    temporary file and ``os.replace``, so concurrent processes (e.g. distributed ranks)
    never see a partially written file.

    Args:
        module_name (str): Dotted name to register, e.g.
            ``"flashcart.o3._tensor_product_kernels"``.
        codegen_file (Path): The ``_codegen_*.py`` module emitting the source.
        kernel_l_max (int): Maximum tensor rank compiled into the generated module.
        build_source (Callable[[int], str]): Generate module source for the requested
            maximum tensor rank.

    Returns:
        ModuleType: Generated module, either already registered in this process or
            imported from the cache.
    """
    module = sys.modules.get(module_name)
    if module is not None:
        return module

    fingerprint = codegen_fingerprint(codegen_file, kernel_l_max)
    path = cache_dir() / f"{module_name.rsplit('.', 1)[-1]}_{fingerprint}.py"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        source = f"{FINGERPRINT_PREFIX}{fingerprint}\n" + build_source(kernel_l_max)
        fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(source)
            os.chmod(tmp_path, 0o644)  # mkstemp creates 0600
            os.replace(tmp_path, path)
        except BaseException:
            Path(tmp_path).unlink(missing_ok=True)
            raise

    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module
