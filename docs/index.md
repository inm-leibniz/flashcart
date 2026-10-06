# Overview

FlashCart provides Cartesian tensor operations and an equivariant interatomic
potential built on them. It includes tools for training and evaluating models
and running simulations with ASE and LAMMPS.

```{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} Installation
:link: installation
:link-type: doc

Install FlashCart and its dependencies.
:::

:::{grid-item-card} Quickstart
:link: quickstart
:link-type: doc

Train, evaluate, and use a FlashCart model.
:::

:::{grid-item-card} API reference
:link: api/index
:link-type: doc

Reference for the Python classes and functions.
:::
```

## Main components

- **Cartesian tensor operations:** Tensor products, tensor expansions, and equivariant
  linear maps, with Triton and PyTorch implementations.
- **FlashCart potential:** An equivariant model built from these tensor operations.
- **Training and evaluation:** Configurable losses, checkpointing, and distributed
  training.
- **Simulation interfaces:** ASE calculators and LAMMPS integration through ML-IAP.

## How it works

FlashCart expresses Cartesian tensor products in the $2l + 1$ independent
components of each rank-$l$ irreducible tensor. Symmetry and trace relations
determine the remaining components, so they need not be stored during model
evaluation. Writing the products in independent components gives polynomial
expressions that can be simplified and differentiated symbolically before
compilation. FlashCart translates these expressions into fused Triton
kernels, including the derivatives required for model evaluation and training.

The FlashCart potential uses these products to describe atomic neighborhoods.
Each message-passing block combines features from neighboring atoms with
tensors describing their relative directions. Summing these contributions
gives an atomic basis, whose repeated tensor products build higher-order
correlations. After each product, a learned linear map mixes the resulting
features back to a fixed number of channels per tensor rank. Each additional
correlation order therefore requires one product and one linear map, without
increasing the channel width. The model predicts atomic energy contributions
from the resulting features and sums them to obtain the total energy.

The tutorials cover {doc}`tutorials/datasets`,
{doc}`tutorials/configs`,
{doc}`tutorials/md`, and
{doc}`tutorials/timing`.
See the {ref}`FlashCart paper <flashcart-paper>` for the model configurations
and benchmark setup.

```{toctree}
:hidden:
:maxdepth: 1
:caption: Getting started

installation
quickstart
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: User guide

guide/configuration
guide/ase
guide/lammps
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Tutorials

tutorials/datasets
tutorials/configs
tutorials/md
tutorials/timing
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Advanced usage

guide/multi_gpu
guide/performance
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Reference

api/index
guide/env_vars
publications
citing
changelog
```
