# LAMMPS interface

FlashCart integrates with LAMMPS through `pair_style mliap unified`. The
`flashcart-lammps` command exports a trained model for this interface. The instructions
below use GPUs with CUDA-enabled Kokkos. LAMMPS must be built with Python support
and use the environment in which FlashCart is installed.

(lammps-export)=
## Export a model

```bash
flashcart-lammps models/my_run          # exports models/my_run/best
flashcart-lammps models/my_run log      # or the latest checkpoint
```

This writes `models/my_run/flashcart-mliap.pt` and prints the element types the model
supports. The exported model is stored on CPU and is moved to the device of the arrays
provided by LAMMPS when first evaluated. Both commands write the same export path,
replacing an existing file. Export again after retraining if you want LAMMPS to use
the updated checkpoint.

The command-line export reports energies relative to the per-element shifts fitted
during training. These constant shifts do not affect forces or energy differences
at fixed composition. To restore the training data's energy reference, construct
{class}`~flashcart.calculators.lammps_mliap.FlashCartLAMMPSUnified` with
`add_atomic_offsets=True` and save it with
{func}`~flashcart.calculators.lammps_mliap.save_lammps_unified`.

(lammps-compile)=
### Compiled evaluation

```bash
flashcart-lammps models/my_run --compile-mode default
```

With `--compile-mode`, FlashCart uses `torch.compile` for energy and force evaluation
in LAMMPS. Compilation starts on the first model evaluation in each LAMMPS process.
Without this option, the model is evaluated without compilation.

The supported modes are `default` and `max-autotune-no-cudagraphs`. Modes using
CUDA graphs are not yet supported by the LAMMPS interface. When constructing
{class}`~flashcart.calculators.lammps_mliap.FlashCartLAMMPSUnified` directly, pass the
mode as `compile_mode`.

Compilation makes the first evaluation slower. Set `TORCHINDUCTOR_CACHE_DIR` to a
persistent directory to reuse cached kernels across runs. Exclude compilation and
warmup from timing measurements. See {doc}`../tutorials/timing` for a timing example.

(lammps-build)=
## Build LAMMPS

If your LAMMPS build already includes the required packages, see
{ref}`lammps-configuration` for the simulation settings.

Before building LAMMPS, activate the Python environment in which you installed
FlashCart. The build requires the `ML-IAP`, `ML-SNAP`, `PYTHON`, and `KOKKOS`
packages, CUDA, Python development headers, and Cython. The Kokkos implementation
of ML-IAP provides the exchange of atomic features between local and ghost atoms
needed by models with multiple interaction layers.
The commands below use CMake:

```bash
pip install cython
git clone -b stable https://github.com/lammps/lammps.git && cd lammps

cmake -S cmake -B build \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_COMPILER=$PWD/lib/kokkos/bin/nvcc_wrapper \
  -D BUILD_SHARED_LIBS=yes -D BUILD_MPI=yes \
  -D PKG_ML-IAP=yes -D PKG_ML-SNAP=yes -D PKG_PYTHON=yes -D MLIAP_ENABLE_PYTHON=yes \
  -D PKG_KOKKOS=yes -D Kokkos_ENABLE_CUDA=yes -D Kokkos_ARCH_AMPERE86=yes \
  -D Python_EXECUTABLE=$(which python)
cmake --build build -j4
cmake --build build --target install-python
pip install cupy-cuda12x      # LAMMPS wraps device arrays with CuPy
```
Replace `Kokkos_ARCH_AMPERE86` with the architecture flag for your GPU.
The CUDA compiler `nvcc` must be on `PATH`. If compilation exhausts available
memory, reduce the number of parallel jobs specified by `-j`.

Put the build on `PATH` and check it:

```bash
export PATH=$PWD/build:$PATH LD_LIBRARY_PATH=$PWD/build:$LD_LIBRARY_PATH
lmp -h | grep -Ei "ML-IAP|KOKKOS"
python -c "import lammps, flashcart"
```

(lammps-configuration)=
## Configure the simulation

The LAMMPS unit convention must match the model's training data. The supplied water
inputs use `units metal`, with eV for energy, Å for distance, and ps for time.

Run the simulation from your working directory. Paths in the LAMMPS input,
including the exported model path below, are relative to that directory.
Adjust them to match your files.

In the LAMMPS input:

```text
pair_style mliap unified models/my_run/flashcart-mliap.pt 0
pair_coeff * * H O
```

`pair_coeff` maps LAMMPS atom types to chemical elements. For a data file with
type 1 = H and type 2 = O, use `* * H O`, even if the model supports additional
elements. The final `0` in `pair_style` excludes ghost atoms as central atoms.
Ghost atoms are copies of neighboring atoms across periodic boundaries or MPI
subdomains. They remain available as neighbors of local atoms.

Each MPI rank must own at least one atom and have at least one neighbor pair.
If some ranks are empty, use fewer ranks or adjust the domain decomposition with
`balance`. Systems with no neighbor pairs are not supported.

## Run the simulation

Use the same input file for one or more GPUs. Run one MPI rank per GPU:

```bash
# one GPU
lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.lammps

# four GPUs, one MPI rank per GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 mpirun -np 4 lmp -k on g 4 -sf kk -pk kokkos newton on neigh half -in in.lammps
```

- Select GPUs with `CUDA_VISIBLE_DEVICES`. Kokkos maps MPI ranks onto the visible
  devices.
- If CPU binding limits performance, adjust the Open MPI `--bind-to` and
  `--map-by` settings to suit the resources allocated to your job.

{doc}`../tutorials/md` demonstrates NVE and NPT simulations of liquid water.

(lammps-troubleshooting)=
## Troubleshooting

Unsupported configurations terminate the LAMMPS process with an error. The MPI
launcher must stop the remaining ranks when this happens. If LAMMPS runs inside
Python, this also exits Python without running cleanup handlers and cannot be
caught as a Python exception.

The following environment variables enable additional logging and checks,
or control CUDA memory-cache handling.

| Variable | Effect |
|---|---|
| `FLASHCART_LAMMPS_DEBUG=1` | Log local and ghost atom counts and index ranges, and validate indices on every call. |
| `FLASHCART_LAMMPS_VALIDATE=1` | Validate neighbor indices on every call (default: first call only). |
| `FLASHCART_LAMMPS_MEMORY_DEBUG=1` | Log CUDA memory around graph conversion, model evaluation, and force copying. |
| `FLASHCART_LAMMPS_EMPTY_CACHE=1` | Release unused cached CUDA memory after force calls. |
| `FLASHCART_LAMMPS_EMPTY_CACHE_EVERY=N` | Release cached memory every N-th force call when `FLASHCART_LAMMPS_EMPTY_CACHE=1`. |

FlashCart sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` unless it is already
defined. This allocator setting can reduce fragmentation when neighbor counts change
during molecular dynamics. It does not limit GPU memory use.

The LAMMPS documentation covers [`pair_style
mliap`](https://docs.lammps.org/pair_mliap.html), [building
ML-IAP](https://docs.lammps.org/Build_extras.html#ml-iap-package), [building
PYTHON](https://docs.lammps.org/Build_extras.html#python-package), and the [KOKKOS
package](https://docs.lammps.org/Speed_kokkos.html).
