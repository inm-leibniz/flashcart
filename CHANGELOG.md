# Changelog

All notable changes to FlashCart are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and FlashCart uses
[semantic versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-10-06

First public release.

### Added

- Generated Triton kernels for Cartesian tensor products, direction-tensor expansions,
  and equivariant linear layers, supporting forward evaluation, backward
  differentiation, and double backward, with a pure PyTorch fallback.
- The FlashCart interatomic potential and a Lightning Fabric trainer with multi-GPU
  support (`flashcart-train`, `flashcart-test`).
- ASE calculator and LAMMPS ML-IAP interface (`flashcart-lammps`).
- Documentation with tutorials and example scripts for dataset preparation, training
  configuration, and molecular dynamics.

[Unreleased]: https://github.com/inm-leibniz/flashcart/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/inm-leibniz/flashcart/releases/tag/v0.1.0
