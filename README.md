<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/inm-leibniz/flashcart/main/docs/_static/flashcart-logo-dark.png">
    <img src="https://raw.githubusercontent.com/inm-leibniz/flashcart/main/docs/_static/flashcart-logo.png" width="420" alt="FlashCart">
  </picture>
</p>

[![Tests](https://github.com/inm-leibniz/flashcart/actions/workflows/unittests.yaml/badge.svg?branch=main&event=push)](https://github.com/inm-leibniz/flashcart/actions/workflows/unittests.yaml)
[![Documentation](https://github.com/inm-leibniz/flashcart/actions/workflows/docs.yaml/badge.svg?branch=main&event=push)](https://inm-leibniz.github.io/flashcart)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/inm-leibniz/flashcart/blob/main/LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2610.06409-b31b1b.svg)](https://arxiv.org/abs/2610.06409)
[![DOI](https://zenodo.org/badge/1405378717.svg)](https://zenodo.org/badge/latestdoi/1405378717)

---

FlashCart provides Cartesian tensor operations and an equivariant interatomic
potential built on them. It derives tensor products and their derivatives
symbolically and generates fused Triton kernels for model evaluation and training.
Cartesian tensor expansions and equivariant linear maps also use generated kernels.
PyTorch implementations support these operations on CPUs and Apple Silicon.

The package also includes:

- the FlashCart potential,
- tools for training and evaluation, including multi-GPU training with Lightning Fabric,
- interfaces for atomistic simulations with ASE and LAMMPS.

See the [documentation](https://inm-leibniz.github.io/flashcart/) for installation,
usage, tutorials, and the Python API.

Trained models accompanying the [FlashCart paper](https://arxiv.org/abs/2610.06409)
are available on
[Zenodo](https://doi.org/10.5281/zenodo.23159584).

## Installation

FlashCart requires Python 3.10 or newer and PyTorch 2.9 or newer. In a virtual
environment, install a PyTorch build for your hardware, then install FlashCart:

```bash
python -m pip install flashcart
```

See [Installation](https://inm-leibniz.github.io/flashcart/installation.html) for
PyTorch installation, hardware support, and installation from source.

## Quickstart

For this example, provide extended XYZ files with reference energies in `REF_energy`
and forces in `REF_forces`. Save the following as `my_run.yaml`. Settings omitted
from the file use the
[defaults](https://inm-leibniz.github.io/flashcart/guide/configuration.html#configuration-defaults):

```yaml
# my_run.yaml
train_path: train.extxyz
valid_path: valid.extxyz
test_path: test.extxyz
output_path: models/my_run
r_max: 5.0
max_epochs: 200
devices: 1
```

Run the commands from the directory containing the data and configuration file.
For this example, use energies in eV, positions in Å, and forces in eV/Å.

The configuration uses one GPU. Select it before training with
`export CUDA_VISIBLE_DEVICES=0`, replacing `0` with the desired GPU index. For a CPU run,
add `device: cpu` to the configuration instead.

```bash
flashcart-train my_run.yaml
flashcart-test my_run.yaml
```

Training saves the best checkpoint in `models/my_run/best`. The evaluation command
uses that checkpoint to evaluate the validation and test sets. See
[Quickstart](https://inm-leibniz.github.io/flashcart/quickstart.html) for data preparation
and an explanation of the output files.

Use the trained model to evaluate a structure from the test set with ASE:

```python
from ase.io import read
from flashcart.calculators import FlashCartCalculator

atoms = read("test.extxyz", index=0)
atoms.calc = FlashCartCalculator(checkpoint="models/my_run/best")
print(atoms.get_potential_energy(), atoms.get_forces().shape)
```

The calculator uses the CPU by default. Pass `device="cuda"` to use the first visible
GPU. Pass `add_atomic_offsets=True` to restore the energy reference used in the
training data. This changes the reported energy but not forces or stress. See
[ASE interface](https://inm-leibniz.github.io/flashcart/guide/ase.html)
for the calculator options.

For LAMMPS, export the model with `flashcart-lammps models/my_run`, then add the
following commands to the input file for a system whose atom types are H and O:

```
pair_style mliap unified models/my_run/flashcart-mliap.pt 0
pair_coeff * * H O
```

The symbols after `pair_coeff * *` map the LAMMPS atom types to chemical elements.
Adjust them to match the atom-type order in your system. Add `--compile-mode default`
to the export command to evaluate the model with `torch.compile` in LAMMPS. See the
[LAMMPS interface](https://inm-leibniz.github.io/flashcart/guide/lammps.html) for the required build
and launch options.

## Learn more

- [Configuration files](https://inm-leibniz.github.io/flashcart/guide/configuration.html)
- [ASE interface](https://inm-leibniz.github.io/flashcart/guide/ase.html)
- [Multi-GPU training](https://inm-leibniz.github.io/flashcart/guide/multi_gpu.html)
- [LAMMPS interface](https://inm-leibniz.github.io/flashcart/guide/lammps.html)
- [Data preparation](https://inm-leibniz.github.io/flashcart/tutorials/datasets.html)
- [Training configurations](https://inm-leibniz.github.io/flashcart/tutorials/configs.html)
- [Molecular dynamics](https://inm-leibniz.github.io/flashcart/tutorials/md.html)
- [Inference and training times](https://inm-leibniz.github.io/flashcart/tutorials/timing.html)
- [Publications](https://inm-leibniz.github.io/flashcart/publications.html)
- [API reference](https://inm-leibniz.github.io/flashcart/api/index.html)

## Citing

If you use FlashCart, please cite the
[FlashCart paper](https://arxiv.org/abs/2610.06409):

```bibtex
@misc{zaverkin2026,
      title={FlashCart: Fast Cartesian Tensor Products for Equivariant Interatomic Potentials},
      author={Viktor Zaverkin and Payman Goodarzi and Sergey V. Sukhomlinov and Davit Hovhannisyan and Roland Aydin and Martin H. Müser and Mathias Niepert},
      year={2026},
      eprint={2610.06409},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2610.06409},
}
```

## License

FlashCart is released under the Apache License 2.0. See
[LICENSE](https://github.com/inm-leibniz/flashcart/blob/main/LICENSE) and
[NOTICE](https://github.com/inm-leibniz/flashcart/blob/main/NOTICE) for the licenses
of adapted third-party code. Contributions are welcome, see
[CONTRIBUTING.md](https://github.com/inm-leibniz/flashcart/blob/main/CONTRIBUTING.md).
