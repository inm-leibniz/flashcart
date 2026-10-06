# Inference and training times

This tutorial shows how to measure inference and training times for a saved
FlashCart model on a fixed atomic structure. It adapts the CUDA-event timing
procedure used for the {ref}`FlashCart paper <flashcart-paper>` to a checkpoint
and structure of your choice.

Inference includes energy and force evaluation. The training measurement
includes prediction, loss evaluation, and differentiation with respect to the
model parameters. The graph is built before timing, and both the graph and
model are transferred to the GPU. Data loading, neighbor-list construction,
device transfers, and optimizer updates are excluded.

## Run the example

The example requires an NVIDIA GPU, a checkpoint in the run's `best/` directory,
and a structure containing elements supported by the checkpoint. Training
measurements also require the run's `config.yaml`. You can reuse the run and
`test.extxyz` from {doc}`../quickstart`. The example uses models trained with
energies in eV and positions in Å. Its synthetic training targets support loss
functions on energies and forces.

Download {download}`measure.py <../../examples/timing/measure.py>` into the
directory containing your training run and structure file, then run:

```bash
CUDA_VISIBLE_DEVICES=0 python measure.py models/my_run test.extxyz
```

The script is also available at `examples/timing/measure.py` in the FlashCart
repository. Paths are resolved from the directory where you
run the command.

Replace `0` with the GPU index for your run. The script reads the first structure
in the file, loads `models/my_run/best`, and measures both inference and training.
Use `--phase inference` or `--phase training` to measure only one phase.
The training loss comes from `models/my_run/config.yaml`.

The script uses float32 with TF32 disabled and keeps the checkpoint's tensor
backend. Predictions are evaluated without `torch.compile` by default. To
measure compiled prediction, run the script again with:

```bash
CUDA_VISIBLE_DEVICES=0 python measure.py models/my_run test.extxyz \
    --compile-mode reduce-overhead
```

Compilation applies to prediction. The training measurement also includes the
loss and its backward pass. Each phase has its own warmup. See
{ref}`kernel-generation-compilation` for the distinction between generated
Triton kernels and `torch.compile`. Use `--compile-mode default` to match the
setting described under {ref}`lammps-compile` in {doc}`../guide/lammps`.
The LAMMPS interface does not support CUDA-graph modes. The inference measurement
excludes neighbor-list construction, communication, and integration, which also
contribute to the time required for a complete simulation.

## What is timed

The same graph is reused for every evaluation. Each measured call performs the
following work:

| Phase | Computation |
|---|---|
| Inference | Evaluate the energy and differentiate it to obtain forces. |
| Training | Clear parameter gradients, evaluate the properties required by the loss, compute the loss, and differentiate it with respect to the model parameters. |

Inference disables parameter gradients but keeps autograd enabled to compute
forces. Training keeps the derivative graph with `create_graph=True` so that
the force loss can be differentiated with respect to model parameters. The
weights remain unchanged because no optimizer update is performed.

Before timing training, the script creates synthetic targets from the model's
predictions plus fixed Gaussian noise. The standard deviation is $10^{-3}N$ eV
for the total energy of an $N$-atom structure and $10^{-2}$ eV/Å for each force
component. The random seed is zero. These targets allow the loss and its
parameter gradients to be computed without reference calculations. They are used
only for timing, not for assessing prediction accuracy. They replace any reference
labels only in the in-memory graph.

## How time is measured

By default, each phase runs 10 untimed warmup evaluations followed by 50 measured
evaluations. CUDA events surround each measured call. Synchronization before
and after the call ensures that earlier GPU work has finished and that the
measurement is complete before its elapsed time is read.

Warmup allows initial compilation and autotuning to take place before timing.
Increase `--warmup` if initial work still affects the measurements, and increase
`--repeat` to collect more samples.

The script reports the mean and sample standard deviation in milliseconds per
evaluation and microseconds per atom. Time per atom is the measured time divided
by the number of atoms in the structure. The standard deviation describes variation
between repeated evaluations of this graph, not variation between training runs.

The script also prints the checkpoint, GPU, software versions, tensor backend,
compilation setting, cutoff, and numbers of atoms and edges.
Keep these settings with the measurements. Time per atom depends on system size
and the number of neighbors. Use the same structure when comparing models.

:::{dropdown} Timing script

```{literalinclude} ../../examples/timing/measure.py
:language: python
```

:::

## The diamond benchmark

The model benchmarks in the {ref}`FlashCart paper <flashcart-paper>` use periodic
diamond cells with a lattice constant of 3.567 Å. For a checkpoint that supports
carbon, prepare a 1000-atom cell with:

```python
from ase.build import bulk
from ase.io import write

atoms = bulk("C", "diamond", a=3.567, cubic=True).repeat((5, 5, 5))
write("diamond.extxyz", atoms)
```

Pass `diamond.extxyz` in place of `test.extxyz`. A `(4, 4, 4)` repeat produces
512 atoms, the size used for the training comparison between tensor backends.
Other benchmarks use different system sizes and compilation settings. To
reproduce a published result, also match its checkpoint, backend, hardware,
and software versions.

## Profiling dataset batches

For operation tables and traces, use {doc}`../guide/performance`.
The `flashcart-profile` command initializes a new model and measures elapsed
wall time over selected dataset batches. This tutorial instead loads a saved
checkpoint and measures individual evaluations of one graph with CUDA events.
Both exclude optimizer updates, but their reported times describe different
workloads and measurement procedures.
