"""Prepare the SPICE (MACE-OFF23) training and test sets.

From the repository root, download and extract the two archives from Cambridge
Apollo (https://www.repository.cam.ac.uk/items/d50227cd-194f-4ba4-aeb7-2643a69f025f):

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

Then run from the repository root:

    python examples/datasets/prepare_spice.py --dir data/spice --seed 0

This writes train_<seed>.extxyz and val_<seed>.extxyz (a random 95/5 split of
the training set), test.extxyz, and one test_subsets/<config_type>.extxyz per
test subset, all under data/spice. Different seeds produce separately sampled
splits, which can overlap.
"""

import argparse
from collections import defaultdict
from pathlib import Path
from random import Random

from ase.io import read, write

VALID_FRACTION = 0.05


def to_reference(frames):
    """Move the calculator's energy and forces to the REF_energy / REF_forces keys FlashCart reads."""
    for atoms in frames:
        atoms.info["REF_energy"] = atoms.get_potential_energy()
        atoms.arrays["REF_forces"] = atoms.get_forces()
        atoms.calc = None
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=Path("."), help="directory with the unpacked .xyz files")
    parser.add_argument("--seed", type=int, default=0, help="seed of the train/validation split")
    args = parser.parse_args()

    train = to_reference(read(args.dir / "train_large_neut_no_bad_clean.xyz", ":"))
    n_valid = round(len(train) * VALID_FRACTION)
    valid_idx = set(Random(args.seed).sample(range(len(train)), n_valid))
    write(args.dir / f"train_{args.seed}.extxyz", [a for i, a in enumerate(train) if i not in valid_idx])
    write(args.dir / f"val_{args.seed}.extxyz", [a for i, a in enumerate(train) if i in valid_idx])
    print(f"seed {args.seed}: {len(train) - n_valid} train / {n_valid} validation structures")

    test = to_reference(read(args.dir / "test_large_neut_all.xyz", ":"))
    write(args.dir / "test.extxyz", test)
    subsets = defaultdict(list)
    for atoms in test:
        subsets[atoms.info["config_type"]].append(atoms)
    (args.dir / "test_subsets").mkdir(exist_ok=True)
    for name, frames in sorted(subsets.items()):
        write(args.dir / "test_subsets" / f"{name.replace(' ', '_')}.extxyz", frames)
        print(f"test subset {name}: {len(frames)} structures")


if __name__ == "__main__":
    main()
