"""Write a toy water dataset (EMT labels) and a tiny CPU config for smoke tests.

    python make_toy_data.py OUT_DIR

Creates OUT_DIR/{train,valid}.extxyz and OUT_DIR/config.yaml; the config trains
a small model for two epochs on CPU into OUT_DIR/model.
"""

import sys
from pathlib import Path

import numpy as np
import yaml
from ase import Atoms
from ase.build import molecule
from ase.calculators.emt import EMT
from ase.io import write


def water_box(rng: np.random.Generator, n_molecules: int = 4, box: float = 5.0) -> Atoms:
    atoms = Atoms(cell=[box] * 3, pbc=True)
    for position in rng.uniform(0.0, box, size=(n_molecules, 3)):
        mol = molecule("H2O")
        mol.euler_rotate(*rng.uniform(0, 360, 3), center="COM")
        mol.translate(position - mol.get_center_of_mass())
        atoms += mol
    atoms.rattle(0.05, seed=int(rng.integers(1 << 31)))
    atoms.wrap()
    atoms.calc = EMT()
    atoms.info["REF_energy"] = atoms.get_potential_energy()
    atoms.arrays["REF_forces"] = atoms.get_forces()
    atoms.calc = None
    return atoms


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    write(out / "train.extxyz", [water_box(rng) for _ in range(32)])
    write(out / "valid.extxyz", [water_box(rng) for _ in range(8)])
    config = dict(
        train_path=str(out / "train.extxyz"),
        valid_path=str(out / "valid.extxyz"),
        test_path=str(out / "valid.extxyz"),
        output_path=str(out / "model"),
        r_max=4.0,
        n_hidden_feats=8,
        n_interactions=1,
        train_batch_size=8,
        eval_batch_size=8,
        max_epochs=2,
        warmup_epochs=1,
        decay_epochs=1,
        weight_decay=1.0e-4,
        device="cpu",
        devices=1,
    )
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"wrote toy data and config to {out}")


if __name__ == "__main__":
    main()
