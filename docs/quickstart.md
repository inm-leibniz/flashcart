# Quickstart

This quickstart shows how to train a FlashCart model, evaluate a saved checkpoint,
and use the model with ASE. Follow {doc}`installation` before starting. You will
also need reference data for training, validation, and testing.

Run the commands from the directory containing your data and configuration file.
Relative paths in the configuration are resolved from this directory.

## Prepare the data

FlashCart reads
[extended XYZ](https://docs.ase-lib.org/ase/io/formatoptions.html#extxyz) files.
This example uses the default training loss, which includes reference energies
and forces. By default, FlashCart reads total energies from
`atoms.info["REF_energy"]` and forces from `atoms.arrays["REF_forces"]`.
See {ref}`configuration-losses` for other training targets.

Use consistent units throughout the dataset. Here, energies are in eV,
positions in Å, and forces in eV/Å. FlashCart does not convert these units.

If ASE reads the reference energies and forces as calculator results, copy them
to these keys before writing the training file. Replace `my_data.xyz` with your
input filename:

```python
from ase.io import read, write

frames = read("my_data.xyz", ":")
for atoms in frames:
    atoms.info["REF_energy"] = atoms.get_potential_energy()
    atoms.arrays["REF_forces"] = atoms.get_forces()
    atoms.calc = None
write("train.extxyz", frames)
```

Convert your validation and test data in the same way, writing `valid.extxyz`
and `test.extxyz`. Use different structures for training, validation, and testing.
The test file is used in the checkpoint evaluation step. It is not required for
training. {doc}`tutorials/datasets` provides scripts
for preparing the datasets used in the {ref}`FlashCart paper <flashcart-paper>`.

(quickstart-training)=
## Train a model

Save the following configuration as `my_run.yaml`. Settings not included in the
file use their default values. See {ref}`configuration-defaults` for the available settings.

```yaml
train_path: train.extxyz
valid_path: valid.extxyz
test_path: test.extxyz
output_path: models/my_run
r_max: 5.0
max_epochs: 200
devices: 1
```

This configuration uses one GPU. For a CPU run, add `device: cpu` to the
configuration. These settings also apply to the evaluation command below.

For an NVIDIA GPU run, use `CUDA_VISIBLE_DEVICES` to select the GPU before
starting training. For example, to select GPU 0:

```bash
export CUDA_VISIBLE_DEVICES=0
```

Replace `0` with the index of the GPU you want to use. With `devices: 1`,
FlashCart uses the first visible GPU, which PyTorch numbers as `cuda:0` within
the process. The exported setting also applies to later commands in the same
shell, including checkpoint evaluation. It is not needed for CPU runs.

Start training with:

```bash
flashcart-train my_run.yaml
```

The run directory `models/my_run` contains:

- `config.yaml`: the configuration used for the run, including defaults and overrides.
- `best/`: the checkpoint with the lowest validation metric `early_stopping_loss`.
- `log/`: the latest saved checkpoint, including the state used to resume training.
- `metrics.csv` and `train.log`: the training metrics and log messages.

The validation metric selects the best checkpoint. Despite its name,
`early_stopping_loss` does not stop training early. Training continues until
`max_epochs`.

Training resumes automatically if `log/training_state.pt` exists in the run
directory. Use a new `output_path` to start a separate run.

At the end of training, FlashCart evaluates the best checkpoint on the training
and validation sets. The next step also evaluates it on the test set.

## Evaluate the checkpoint

```bash
flashcart-test my_run.yaml
```

This evaluates `models/my_run/best` on the validation and test sets and writes
`models/my_run/test.log`. Pass `checkpoint=...` to evaluate another checkpoint
or `test_path=...` to select another test set. Add `valid_path=null` to evaluate
only the test set.

The evaluation command reads device settings from `my_run.yaml` and its own
command-line overrides. Overrides passed to the training command do not carry over.

## Use the model with ASE

Use the trained model to evaluate the first structure in the test file:

```python
from ase.io import read
from flashcart.calculators import FlashCartCalculator

atoms = read("test.extxyz", index=0)
atoms.calc = FlashCartCalculator(checkpoint="models/my_run/best")
print(atoms.get_potential_energy(), atoms.get_forces().shape)
```

The structure must contain only elements supported by the checkpoint.
The calculator loads the model on the CPU by default. Pass `device="cuda"`
to use a GPU.

By default, the calculator does not add the per-element energy shifts fitted
during training. Pass `add_atomic_offsets=True` to report energies with the same
reference as the training data. The shifts do not affect forces.

## Next steps

To export the best checkpoint for LAMMPS, run:

```bash
flashcart-lammps models/my_run
```

This writes `models/my_run/flashcart-mliap.pt` and prints the elements supported by
the model. Running the exported model requires a LAMMPS build with Python-enabled
ML-IAP. See {doc}`guide/lammps` for setup and input commands.

See {doc}`guide/ase` for calculator options and {doc}`tutorials/md` for a
simulation example. Command-line overrides are described in
{ref}`configuration-overrides`.
For training on several GPUs, see {doc}`guide/multi_gpu`.
