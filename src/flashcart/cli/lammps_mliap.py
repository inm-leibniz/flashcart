"""Export a training run for LAMMPS ML-IAP (the ``flashcart-lammps`` script)."""

import argparse
from pathlib import Path


def _checkpoint_dir(model_path: Path, checkpoint: str) -> Path:
    """Resolve and check the selected checkpoint directory.

    Args:
        model_path (Path): Training output directory.
        checkpoint (str): Checkpoint subdirectory, usually ``"best"`` or ``"log"``.

    Returns:
        Path: Existing checkpoint directory.

    Raises:
        FileNotFoundError: The selected checkpoint directory does not exist.
    """
    checkpoint_dir = model_path / checkpoint
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    return checkpoint_dir


def _output_path(model_path: Path) -> Path:
    """Return the ML-IAP export path within a training directory.

    Args:
        model_path (Path): Training output directory.

    Returns:
        Path: Path to ``flashcart-mliap.pt`` in the training directory.
    """
    return model_path / "flashcart-mliap.pt"


def main() -> None:
    """Export a run checkpoint for the LAMMPS ML-IAP interface.

    ``flashcart-lammps RUN_DIR [best|log] [--compile-mode MODE]`` reads the selected
    checkpoint, using ``best`` by default, and writes ``RUN_DIR/flashcart-mliap.pt``.
    The command prints the supported chemical elements. In the LAMMPS input,
    ``pair_coeff`` maps each LAMMPS atom type to one of these elements.

    With ``--compile-mode``, the export stores the selected compilation setting.
    FlashCart uses ``torch.compile`` when the model is first evaluated in LAMMPS.
    Modes using CUDA graphs are not supported by this interface. Without this option,
    evaluation is uncompiled. Exporting replaces an existing output file.
    """
    from flashcart.calculators.lammps_mliap import COMPILE_MODES, build_lammps_unified, save_lammps_unified

    parser = argparse.ArgumentParser(description="Export a FlashCart run directory for LAMMPS ML-IAP unified.")
    parser.add_argument(
        "model_path",
        type=Path,
        help="Training output directory.",
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        choices=("best", "log"),
        default="best",
        help="Checkpoint subdirectory (default: best).",
    )
    parser.add_argument(
        "--compile-mode",
        choices=COMPILE_MODES,
        default=None,
        help="Compile the model with torch.compile inside LAMMPS on its first evaluation. "
        "Compilation adds an initial cost (default: disabled).",
    )
    args = parser.parse_args()

    try:
        checkpoint_dir = _checkpoint_dir(args.model_path, args.checkpoint)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    output = _output_path(args.model_path)
    unified = build_lammps_unified(checkpoint_dir, compile_mode=args.compile_mode)
    output = save_lammps_unified(unified, output)
    print(f"Wrote ML-IAP model: {output}")
    print(f"Source checkpoint: {checkpoint_dir}")
    print("Element types: " + " ".join(unified.element_types))
    if args.compile_mode is not None:
        print(
            f"Compile mode: {args.compile_mode}. FlashCart compiles the model on its first evaluation in LAMMPS. "
            "Set TORCHINDUCTOR_CACHE_DIR to a persistent directory to reuse compiled kernels across runs."
        )


if __name__ == "__main__":
    main()
