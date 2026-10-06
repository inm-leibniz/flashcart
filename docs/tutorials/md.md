# Molecular dynamics

This tutorial uses liquid water to illustrate molecular dynamics with a trained
FlashCart potential in LAMMPS. It examines energy conservation and density in a
periodic system of 256 water molecules. The simulations use Kokkos on one GPU
and a time step of 0.5 fs. The two calculations probe different properties:

- **Energy conservation:** 50 ps of NVT equilibration at 300 K, followed by 1.2 ns of
  NVE dynamics. The analysis reports the drift and fluctuation of total energy over the
  final 1 ns.
- **Density:** 1.2 ns of NPT dynamics at 1 bar, at temperatures from 273.15 K to 373.15
  K in 20 K steps. The analysis reports the mean density and its block-averaged standard
  error over the final 1 ns.

## Run the simulations

The input files and scripts are in the repository's `examples/md/water` directory.
Use a LAMMPS build with ML-IAP, Python support, and CUDA-enabled Kokkos,
available as `lmp` on `PATH`. See {ref}`lammps-build` for build instructions.

The model must support hydrogen and oxygen and use eV for energy and Å for
distance. It should describe water over the temperatures and densities studied.

From the repository root, select a GPU and run:

```bash
CUDA_VISIBLE_DEVICES=0 bash examples/md/water/run.sh models/my_run
```

Replace `0` with the GPU index for your run. This command runs the complete
example: NVT equilibration followed by one NVE trajectory, then six NPT
trajectories. The simulations run one after another on the GPU you specify.

The script creates the initial box and exports the model with
`flashcart-lammps models/my_run --compile-mode default` if those files do not
already exist. Existing model exports are reused. If the checkpoint or compilation
setting has changed, run the export again before starting:

```bash
flashcart-lammps models/my_run --compile-mode default
```

With compilation enabled, the model is compiled when first evaluated in each LAMMPS
process. See {ref}`lammps-compile` in {doc}`../guide/lammps`.

Logs, trajectories, and saved structures are written to
`examples/md/water/results/my_run`. The analysis prints its summary to the
terminal. The directory name comes from the final component of the training
directory path. Runs with the same name reuse this output directory and start
the simulations again.

Energy conservation tests the force evaluation and time integration. It does
not by itself establish the accuracy of the potential.

:::{dropdown} Simulation script

```{literalinclude} ../../examples/md/water/run.sh
:language: bash
```

:::

## Build the box

The builder places 256 water molecules (768 atoms) in a periodic cube of side
19.706 Å, with random molecular orientations and a fixed seed. It writes a LAMMPS
data file and an extended XYZ file. The initial box is unrelaxed. Both LAMMPS inputs
minimize it before assigning velocities and starting dynamics.

:::{dropdown} Water box construction script

```{literalinclude} ../../examples/md/water/build_box.py
:language: python
```

:::

## LAMMPS inputs

::::{tab-set}
:::{tab-item} in.nve
```{literalinclude} ../../examples/md/water/in.nve
:language: text
```
:::
:::{tab-item} in.npt
```{literalinclude} ../../examples/md/water/in.npt
:language: text
```
:::
::::

## Analysis

The analysis reports the slope of a linear fit to total energy as the energy
drift in meV/atom/ns. Energy fluctuations are the standard deviation of total
energy per atom in meV/atom, without subtracting the fitted drift.

For NPT trajectories, the analysis reports the mean density in g/cm³ and
estimates its standard error from 50 ps block averages. The standard error
quantifies uncertainty in the mean density due to the finite trajectory length.
Compare densities with experimental values at matching temperatures and
pressure. Simulation speed in ns/day is taken from the LAMMPS performance
summary for the production run.

To repeat the analysis without rerunning the simulations:

```bash
python examples/md/water/analyze.py examples/md/water/results/my_run \
    --analysis-ps 1000 --block-ps 50
```

The default atom count is 768. If you change the box size, pass `--n-atoms` so that
the energy drift and fluctuation are divided by the correct number of atoms.

:::{dropdown} Analysis script

```{literalinclude} ../../examples/md/water/analyze.py
:language: python
```

:::

For molecular dynamics with ASE, see {ref}`ase-molecular-dynamics` in
{doc}`../guide/ase`.
