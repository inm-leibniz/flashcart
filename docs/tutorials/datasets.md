# Data preparation

This tutorial prepares the SPICE (MACE-OFF23) and MAD datasets for the examples
in {doc}`configs`. The scripts read energies and forces with ASE, store them
under the reference keys used by FlashCart, and write extended XYZ files.
They preserve the source units: eV for energies, Å for positions, and eV/Å for
forces. No unit conversion is applied.

By default, FlashCart reads total reference energies from
`atoms.info["REF_energy"]` and forces from `atoms.arrays["REF_forces"]`.
Set `energy_key` and `forces_key` to use other names.

These examples use scripts from the cloned FlashCart repository. See
{doc}`../installation` for installation instructions. Run the commands from
the repository root, with `wget` and `tar` available for downloading and
extracting the data.

## SPICE (MACE-OFF23)

Download the training and test archives from the
[Cambridge Apollo repository](https://www.repository.cam.ac.uk/items/d50227cd-194f-4ba4-aeb7-2643a69f025f)
and extract them into `data/spice`:

```bash
mkdir -p data/spice
(
    cd data/spice
    wget --user-agent="Mozilla/5.0" -O train.tar.gz \
        "https://www.repository.cam.ac.uk/bitstreams/b185b5ab-91cf-489a-9302-63bfac42824a/download"
    wget --user-agent="Mozilla/5.0" -O test.tar.gz \
        "https://www.repository.cam.ac.uk/bitstreams/cb8351dd-f09c-413f-921c-67a702a7f0c5/download"
    tar xzf train.tar.gz
    tar xzf test.tar.gz
)
```

The preparation script expects `train_large_neut_no_bad_clean.xyz` and
`test_large_neut_all.xyz` directly in `data/spice`. The commands in parentheses
run in that directory and leave the current shell in the repository root.

The script divides the supplied training set randomly into 95% training and
5% validation data. `--seed` controls the split. Different seeds produce
separately sampled splits, which can overlap. The supplied test set is shared
by all seeds.

Prepare the three splits used by the SPICE model configurations:

```bash
for seed in 0 1 2
do
    python examples/datasets/prepare_spice.py --dir data/spice --seed "$seed"
done
```

Each call writes `train_<seed>.extxyz` and `val_<seed>.extxyz` in `data/spice`.
It also writes `test.extxyz` and groups the test structures by `config_type`
into files under `test_subsets`. These test files are rewritten from the same
supplied test set on each call.

:::{dropdown} SPICE preparation script

```{literalinclude} ../../examples/datasets/prepare_spice.py
:language: python
```

:::

## MAD

Download the supplied training, validation, and test splits from
[Materials Cloud](https://archive.materialscloud.org/records/c4ene-0mv14),
and the benchmark structures from the
[PET-MAD repository](https://huggingface.co/lab-cosmo/pet-mad/tree/main/benchmarks):

```bash
mkdir -p data/mad
for split in train val test
do
    wget -O "data/mad/mad-$split.xyz" \
        "https://archive.materialscloud.org/records/c4ene-0mv14/files/mad-$split.xyz?download=1"
done
wget -O data/mad/mad-bench.xyz \
    "https://huggingface.co/lab-cosmo/pet-mad/resolve/main/benchmarks/mad-test-mad-settings.xyz"
```

The preparation script expects `mad-train.xyz`, `mad-val.xyz`, `mad-test.xyz`,
and `mad-bench.xyz` in `data/mad`. It preserves the supplied splits and groups
the MAD test set by `subset` and the PET-MAD benchmark by `dataset`.
Structures without the relevant key are assigned to an `unlabelled` group.

Run the preparation script:

```bash
python examples/datasets/prepare_mad.py --dir data/mad
```

This writes `mad-train.extxyz`, `mad-val.extxyz`, `mad-test.extxyz`, and
`mad-bench.extxyz` in `data/mad`, together with the grouped files in
`data/mad/subsets`. The reference energies already have the isolated-atom
baseline subtracted. Keep that reference when training with the supplied MAD
configuration.

:::{dropdown} MAD preparation script

```{literalinclude} ../../examples/datasets/prepare_mad.py
:language: python
```

:::
