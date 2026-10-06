# Liquid water with LAMMPS

This example measures energy conservation in the microcanonical ensemble (NVE) and
density in the isothermal-isobaric ensemble (NPT) for a periodic box of 256 water
molecules. Each production trajectory runs for 1.2 ns with a time step of 0.5 fs.

The example requires a LAMMPS build with ML-IAP, Python support, and CUDA-enabled
Kokkos, available as `lmp` on `PATH`. The model must support hydrogen and oxygen,
use eV for energy and Å for distance, and describe water over the temperatures
and densities studied. The SPICE training configurations in
[`examples/configs`](../../configs) provide one starting point.

Run the following command from this directory, replacing the argument with the path
to your training output directory:

```bash
CUDA_VISIBLE_DEVICES=0 bash run.sh ../../../models/my_run
```

The script builds the initial box with `build_box.py` and exports the trained model
with `flashcart-lammps ../../../models/my_run --compile-mode default` if these
files do not already exist. It then runs one NVE trajectory, preceded by 50 ps of
NVT equilibration at 300 K, and six NPT trajectories at 1 bar, with temperatures
from 273.15 to 373.15 K in steps of 20 K.

Finally, `analyze.py` prints the NVE energy drift in meV/atom/ns, energy
fluctuations in meV/atom, NPT densities with block-averaged standard errors, and
simulation speeds. By default, it analyzes the final 1 ns of each trajectory.
Logs, trajectories, and saved structures are written to
`results/<training-directory-name>`. Runs with the same name reuse this output
directory and start the simulations again.

Existing model exports are reused. If the checkpoint or compilation setting has
changed, export the model again before starting:

```bash
flashcart-lammps ../../../models/my_run --compile-mode default
```

With compilation enabled, the model is compiled when first evaluated in each LAMMPS
process.

`run.sh` runs the trajectories sequentially on one GPU. Each simulation starts
from the initial water box and can also be launched separately using the
commands in `in.nve` and `in.npt`. Select the GPU with `CUDA_VISIBLE_DEVICES`
and use distinct output directories and log paths.
