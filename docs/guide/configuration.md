# Configuration files

Training, evaluation, and profiling read their settings from YAML files and
command-line overrides.

(configuration-overrides)=
## Configuration files and overrides

The training, evaluation, and profiling commands accept an optional YAML file followed
by `KEY=VALUE` overrides:

```bash
flashcart-train my_run.yaml lr=2e-3 devices=2
```

Training and evaluation assemble the configuration in three layers:

1. Packaged defaults from `flashcart/configs/default.yaml`, listed in
   {ref}`configuration-defaults`.
2. Values from the supplied YAML file.
3. Command-line overrides.

Each layer replaces matching top-level values. Nested mappings and lists are replaced as
a whole. A configuration therefore needs only the keys that differ from the defaults.
Profiling also loads the defaults in `flashcart/configs/profile.yaml` before applying
the supplied file and command-line overrides.

Overrides accept numbers, booleans, null values, and YAML lists or mappings.
Quote values containing spaces or shell-sensitive characters, for example
`elements='[H, O]'`. Training saves the resulting configuration as `config.yaml`
in the run directory.

```{note}
FlashCart does not reject unrecognized configuration keys. A misspelled key can
therefore be ignored without an error. Check key names in
{ref}`configuration-defaults`.
```

## Data and output paths

Training requires `train_path`, `valid_path`, and `output_path`. The data paths
point to extended XYZ files. See {doc}`../tutorials/datasets` for the reference keys
and units. `output_path` sets the run directory. Relative paths start from the
working directory where you run the command, not the directory containing the
YAML file.

Evaluation requires either `checkpoint` or `output_path`, together with at least
one of `valid_path` and `test_path`. If `checkpoint` is omitted, `flashcart-test`
loads `output_path/best`. The model architecture comes from the checkpoint's
`model_config.yaml`. The evaluation configuration supplies data paths, metrics,
and execution settings.

Profiling uses `profile_data_path` and `profile_train_path`, with `train_path` as a
fallback, and does not require `output_path`. See {ref}`profiling-configuration`
for the profiling options. Model export uses a run directory instead of a
configuration file. See {ref}`lammps-export` for the export command.

## Initializing and resuming training

If `output_path/log/training_state.pt` exists, training resumes from that saved
state, including the optimizer, scheduler, epoch, and exponential moving average
(EMA) state when available. The `checkpoint` setting is ignored when this saved
state is present. FlashCart still uses the current configuration to construct
the model and training task, so keep its architecture and optimizer settings
compatible with the saved state.

Otherwise, `checkpoint` initializes model weights for a new run. Use a new
`output_path` when starting a separate run, and match the model architecture and
element order to those of the checkpoint. See {ref}`quickstart-training` in
{doc}`../quickstart` for the files written during training.

(configuration-losses)=
## Loss functions

`train_loss` defines the objective used to update model parameters.
`early_stopping_loss` defines the validation metric used to select the best
checkpoint. `valid_losses` lists the metrics reported during training and
checkpoint evaluation.

Both loss settings and each entry of `valid_losses` accept a single property
term or a `weighted_sum` of terms. The fields below define each term:

| Field | Values |
|---|---|
| `property` | `energy`, `forces`, `stress`, `virials` |
| `reduce` | `mae`, `mse`, `rmse`, `sae`, `sse`, `maxe`, `huber_mean`, `huber_sum`, `norm_mean`, `norm_sum` |
| `normalization` | `per_atom`, `per_sqrt_atom`, `none` |
| `delta` | Huber threshold, used with `huber_*` |
| `weight` | Term weight inside `weighted_sum` |

Normalization acts on the prediction error before the reduction. This error,
called the residual, is the prediction minus the reference value. For energy,
`per_atom` divides each structure's energy error by its atom count, and
`per_sqrt_atom` divides it by the square root of that count. Forces are already
specified per atom and require `normalization: none`. Stress also requires
`normalization: none` because it is already normalized by volume.

Stress and virial losses require reference stresses and cells with nonzero
volume. The command-line data loader reads stresses from `atoms.info["REF_stress"]`
and derives reference virials as `-stress * volume`. Both quantities use the
six-component Voigt order `(xx, yy, zz, yz, xz, xy)`.

For force errors, `mae`, `mse`, and `rmse` average over all Cartesian components.
`norm_mean` instead averages the lengths of the force-error vectors over atoms.
`norm_sum` sums those lengths. The norm includes a small term, `1e-12`, under the
square root. The sum reductions (`sae`, `sse`, `huber_sum`, and `norm_sum`) grow with
the number of observations, so their scale depends on batch size. Huber reductions
require a positive `delta`, expressed in the units of the normalized residual.

For example, this metric measures the energy MAE per atom:

```yaml
early_stopping_loss:
  property: energy
  reduce: mae
  normalization: per_atom
```

The default `train_loss` is a weighted sum of Huber losses. The example
configurations from the {ref}`FlashCart paper <flashcart-paper>` instead combine
per-atom energy MAE with the mean norm of force-error vectors (`norm_mean`). See
{doc}`../tutorials/configs` for their terms and weights.

By default, `early_stopping_loss` combines per-atom energy MAE with componentwise
force MAE. Despite its name, it does not stop training when the metric stops
improving. Training continues to `max_epochs`.

(configuration-defaults)=
## Default configuration

```{literalinclude} ../../src/flashcart/configs/default.yaml
:language: yaml
```
