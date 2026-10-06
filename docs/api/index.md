# API reference

Use the Python API to construct FlashCart models, evaluate trained potentials,
and work directly with Cartesian tensor operations.
See {doc}`../quickstart` for a complete training and evaluation example.

| Package | Description |
|---|---|
| {doc}`model` | FlashCart potential and the base class for atomistic models. |
| {doc}`calculators` | Interfaces to atomistic simulation software, including ASE and LAMMPS. |
| {doc}`o3` | Irreducible Cartesian tensors, tensor products, and equivariant linear maps. |
| {doc}`nn` | Layers and radial functions for equivariant message passing. |
| {doc}`data` | Atomic configurations, datasets, neighbor graphs, and batching. |
| {doc}`training` | Model training and evaluation, loss functions, optimizers, and learning-rate schedules. |
| {doc}`cli` | Command-line entry points for training, evaluation, profiling, and model export. |
| {doc}`utils` | Shared helpers for configuration, geometry, distributed execution, and logging. |

```{toctree}
:hidden:
:maxdepth: 2

model
calculators
o3
nn
data
training
cli
utils
```
