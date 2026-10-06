"""Prepare the MAD dataset and the PET-MAD benchmark.

From the repository root, download the official splits from Materials Cloud
(https://archive.materialscloud.org/records/c4ene-0mv14) and the 1562-structure
benchmark shipped with PET-MAD (Table 1 of arXiv:2503.14118):

    mkdir -p data/mad
    for split in train val test
    do
        wget -O "data/mad/mad-$split.xyz" \
            "https://archive.materialscloud.org/records/c4ene-0mv14/files/mad-$split.xyz?download=1"
    done
    wget -O data/mad/mad-bench.xyz \
        "https://huggingface.co/lab-cosmo/pet-mad/resolve/main/benchmarks/mad-test-mad-settings.xyz"

Then run from the repository root:

    python examples/datasets/prepare_mad.py --dir data/mad

This writes mad-{train,val,test,bench}.extxyz and one
subsets/<file>-<group>.extxyz per MAD test subset and per benchmark source
dataset, all under data/mad. The reference energies already have the
isolated-atom baseline subtracted. The training configuration starts from
zero prior offsets and fits the remaining per-element energy shifts.
"""

import argparse
from collections import defaultdict
from pathlib import Path

from ase.io import read, write

GROUP_KEYS = {"mad-test": "subset", "mad-bench": "dataset"}


def to_reference(frames):
    """Move the calculator's energy and forces to the REF_energy / REF_forces keys FlashCart reads."""
    for atoms in frames:
        atoms.info["REF_energy"] = atoms.get_potential_energy()
        atoms.arrays["REF_forces"] = atoms.get_forces()
        atoms.calc = None
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=Path("."), help="directory with the downloaded .xyz files")
    args = parser.parse_args()

    (args.dir / "subsets").mkdir(exist_ok=True)
    for name in ("mad-train", "mad-val", "mad-test", "mad-bench"):
        frames = to_reference(read(args.dir / f"{name}.xyz", ":"))
        write(args.dir / f"{name}.extxyz", frames)
        print(f"{name}: {len(frames)} structures")
        if name not in GROUP_KEYS:
            continue
        groups = defaultdict(list)
        for atoms in frames:
            groups[str(atoms.info.get(GROUP_KEYS[name], "unlabelled")).replace("/", "_")].append(atoms)
        for group, members in sorted(groups.items()):
            write(args.dir / "subsets" / f"{name}-{group}.extxyz", members)
            print(f"  {group}: {len(members)} structures")


if __name__ == "__main__":
    main()
