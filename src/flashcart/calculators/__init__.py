"""Interfaces for atomistic simulations with trained FlashCart models.

The package exposes ``FlashCartCalculator`` for use with the Atomic Simulation
Environment (ASE). The LAMMPS ML-IAP interface is available separately through
``flashcart.calculators.lammps_mliap``. Importing that module sets a default PyTorch
CUDA allocator configuration and attempts to load the optional LAMMPS dependency.
Keeping this import separate avoids these effects when using the ASE calculator.
"""

from flashcart.calculators.ase import FlashCartCalculator

__all__ = ["FlashCartCalculator"]
