# Training configurations

This tutorial trains models using the configurations from the
{ref}`FlashCart paper <flashcart-paper>` and shows how to generate configurations
for the five SPICE models, each with three random seeds. For the configuration
format and available settings, see {doc}`../guide/configuration`.

The examples use files from the cloned FlashCart repository. Follow
{doc}`../installation` and {doc}`datasets` before starting, then run the commands
from the repository root.

Select the GPU before training. The following settings also disable TF32 to
match the numerical settings used in the {ref}`FlashCart paper <flashcart-paper>`:

```bash
export CUDA_VISIBLE_DEVICES=0
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
```

Replace `0` with the GPU index for your run. These settings apply to later
commands in the same shell.

## Train a SPICE model

`examples/configs/spice.yaml` defines a model with approximately 1.1 million
parameters, three interactions, correlation order three, and 64 feature channels.
Train and evaluate it with:

```bash
bash examples/configs/train_and_test.sh examples/configs/spice.yaml data/spice/test_subsets
```

The script trains one model, evaluates its best checkpoint on the configured
test set, and then evaluates each supplied test subset. This configuration
writes the run to `models/i3c3f64l2_s0`. The evaluation produces `test.log`
and a separate log for each subset under `test_subsets` in that run directory.

```{literalinclude} ../../examples/configs/spice.yaml
:language: yaml
```

:::{dropdown} Training and evaluation script

```{literalinclude} ../../examples/configs/train_and_test.sh
:language: bash
```

:::

## Generate model configurations

`make_configs.py` generates the five SPICE model configurations reported in the
{ref}`FlashCart paper <flashcart-paper>`, each with seeds 0, 1, and 2:

| Model | Configuration name before the seed suffix | GPUs |
|---|---|---|
| FlashCart-69k | `i2c3f20l2r` | 1 |
| FlashCart-0.5M | `i3c5f32l2` | 1 |
| FlashCart-1.1M | `i3c3f64l2` | 1 |
| FlashCart-3.0M | `i3c5f88l2` | 1 |
| FlashCart-5.6M | `i4c5f104l2` | 2 |

The script updates the architecture, seeds, data paths, output directory,
and GPU count for each run. Generate the files with:

```bash
python examples/configs/make_configs.py --out configs
```

This writes 15 configuration files without launching training. To train a
model, pass one generated file to the training script. For example, this
command selects the same architecture as above with seed 1:

```bash
bash examples/configs/train_and_test.sh configs/i3c3f64l2_s1.yaml data/spice/test_subsets
```

Each generated configuration has a distinct output directory. Repeat the
command with the other configurations you want to evaluate.

FlashCart-5.6M requests two GPUs. Make two devices available with a setting
such as `CUDA_VISIBLE_DEVICES=0,1` before running the command.
The global batch target is unchanged, but actual batches can differ under
dynamic batching. See {ref}`distributed-batching` for how batches are
distributed across GPUs.

:::{dropdown} Configuration generation script

```{literalinclude} ../../examples/configs/make_configs.py
:language: python
```

:::

## Train a MAD model

`examples/configs/mad.yaml` uses a smaller cutoff and gives force errors a lower
weight in the training loss function than the SPICE configuration. MAD already
subtracts an isolated-atom baseline from its reference energies. The supplied
configuration keeps `atomic_shifts: ~`, so FlashCart starts from zero prior
offsets and fits the remaining per-element energy shifts.

Train and evaluate the three seeds, each with a separate output directory:

```bash
for seed in 0 1 2
do
    flashcart-train examples/configs/mad.yaml model_seed="$seed" data_seed="$seed" \
        output_path="models/mad_i3c3f96l2_s${seed}"
    flashcart-test examples/configs/mad.yaml model_seed="$seed" data_seed="$seed" \
        output_path="models/mad_i3c3f96l2_s${seed}"
done
```

MAD uses fixed dataset splits. Here, `data_seed` changes the training shuffle.
The configuration uses one GPU, selected from the visible devices.

```{literalinclude} ../../examples/configs/mad.yaml
:language: yaml
```

## Evaluate and compare models

To evaluate the seed-zero MAD model on the complete PET-MAD benchmark, run:

```bash
flashcart-test examples/configs/mad.yaml valid_path=null \
    test_path=data/mad/mad-bench.extxyz output_path=models/mad_i3c3f96l2_s0
```

This reports errors over the complete benchmark. To obtain results for
individual benchmark datasets, evaluate the corresponding `mad-bench-*.extxyz`
files in `data/mad/subsets`, passing one filename as `test_path` for each run.
Use the matching `output_path` when evaluating another model seed.

The {ref}`FlashCart paper <flashcart-paper>` reports SPICE errors averaged
equally over the seven test subsets, followed by the mean and standard deviation
across three training seeds.
