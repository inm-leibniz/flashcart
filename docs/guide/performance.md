# Performance and profiling

Use profiling to compare model configurations and identify operations that
account for most of the evaluation time. This page also explains the compilation
and caching behavior that can make initial evaluations slower than later ones.
For timing measurements of a saved checkpoint, see
{doc}`../tutorials/timing`.

(kernel-generation-compilation)=
## Kernel generation and compilation

FlashCart evaluates tensor products, irreducible Cartesian expansions, and equivariant
linear maps with generated Triton kernels when `use_triton: true` and the inputs
are on CUDA. The kernels combine operations to reduce intermediate storage and
memory traffic. Their generated derivatives support training on energy derivatives.
Tensor-product evaluation can use several launches, grouped by output tensor rank
or requested derivatives.

`FLASHCART_KERNEL_L_MAX` sets the maximum tensor rank included in generated modules
(default: 3). Set it before importing FlashCart if the model requires higher ranks.
Increasing the limit adds symbolic expressions and can increase generation and
compilation time. Each model evaluates only its requested tensor ranks.

The generated source files are cached when their backends are first imported.
Later imports reuse these files. Updating FlashCart or changing the rank limit
can require different source files, which are generated automatically if they
are not already cached. See {doc}`env_vars` for the cache location and settings.

`predict_compile: true` enables `torch.compile` for predictions during training,
validation, and testing. It also pads batches to allocated capacities, reducing
changes in input shape while batches fit those capacities. Larger batches can
increase the capacities and require recompilation. The `predict_pad_*_multiple`
options control how capacities are rounded. Larger multiples can reduce shape
changes while increasing the padded work and memory use.

Kernel compilation and autotuning, as well as `torch.compile` when enabled, add
initial overhead. Include untimed warmup runs before measuring repeated
predictions, and use the same structures, cutoff, model, and precision settings
when comparing backends.

(profiling-command)=
## Run the profiler

The profiler initializes a model from the configuration rather than loading
a checkpoint. By default, it also fits the data-dependent scales and offsets.
Both steps take place before timing. Set `profile_pre_fit=false` to skip fitting.

Profiling uses one device. Select the GPU with `CUDA_VISIBLE_DEVICES` before
running the command. For example, to use GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 flashcart-profile my_run.yaml device=gpu \
    profile_phase=both profile_batch_size=32 profile_n_batches=2 \
    profile_repeat=5 profile_trace_dir=traces/my_run
```

Replace `1` with the GPU index for your run. The selected GPU is the only one
visible to the process and is numbered `cuda:0` within it. For CPU profiling,
use `device=cpu` and omit `CUDA_VISIBLE_DEVICES`.

Model and data settings come from the supplied configuration. The `profile_*`
options control the measurements and are listed in
{ref}`profiling-configuration`. `profile_data_path` selects the inference data,
falling back to `train_path`. `profile_train_path` selects the training data,
falling back to `train_path` and then
`profile_data_path`.

The `devices`, `n_nodes`, and `strategy` settings do not launch distributed
profiling. The profiler also does not apply Fabric's `precision` setting.

The profiler's `profile_compile_predict` option controls compiled predictions
independently of the training and evaluation option `predict_compile`. The
profiler does not pad batches as those commands do. Its default is fixed-size
batching. Set `profile_dynamic_batching=true` to calculate atom and edge budgets
from `profile_batch_size`.

## What is measured

`flashcart-profile` measures inference and the computations used during training.
Inference evaluates energies and forces. Training evaluates the predictions
required by the configured loss, computes that loss, and differentiates it with
respect to the model parameters.

Data loading, neighbor-list construction, and transfers to the device take place
before timing. Optimizer updates are excluded. After untimed warmup runs, the
same batches are evaluated repeatedly. CUDA execution is synchronized before and
after timing. The reported `s/run` is the average time to process all batches
included in the measurement.

Use the `[timer]` output for elapsed time. The operation tables help identify
expensive operations but include profiler overhead. These measurements describe
model computation. They do not measure a complete training iteration or simulation.

The runtime measurements in the {ref}`FlashCart paper <flashcart-paper>` use
float32 with TF32 disabled. `flashcart-profile` does not set the TF32 options
automatically. The repository's `examples/configs/train_and_test.sh` shows
the environment settings used by the training example. Record the precision
settings when comparing runtimes or prediction errors.

## Read the measurements

Each run processes the batches selected by `profile_n_batches`. Use the printed
structure, atom, and edge counts to interpret throughput. The `[phase-timer]`
lines separately measure training prediction, loss evaluation, and parameter
differentiation. These measurements synchronize CUDA after each phase and can
differ from the combined `[timer]` measurement.

`profile_warmup` controls untimed runs and `profile_repeat` controls measured
repetitions. Increase these when initial compilation or variation between runs
affects the measurement.

Setting `profile_trace_dir` exports Chrome traces that can be opened in
`chrome://tracing` or [Perfetto](https://ui.perfetto.dev). Profiler overhead can be
substantial for many small operations, so use these traces to examine the work
performed and the `[timer]` lines to compare elapsed times.

(profiling-configuration)=
## Profiling configuration

```{literalinclude} ../../src/flashcart/configs/profile.yaml
:language: yaml
```
