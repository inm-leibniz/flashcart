"""Build the 256-molecule water box used for the MD runs.

    python build_box.py

Places H2O molecules with random orientations (seed 0) on the FCC sites of a
4x4x4 grid in a 19.706 Å periodic cube (about 1 g/cm^3) and writes
water_256.data (LAMMPS) and water_256.extxyz (ASE). The box is not relaxed.
in.nve and in.npt minimize it first.
"""

from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.io import write

BOX = 19.706
FCC = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
CELLS = 4
SEED = 0
OUT = Path("water_256")


def build():
    rng = np.random.default_rng(SEED)
    a = BOX / CELLS
    box = Atoms(cell=[BOX] * 3, pbc=True)
    for i, j, k in np.ndindex(CELLS, CELLS, CELLS):
        for site in FCC:
            mol = molecule("H2O")
            mol.euler_rotate(*rng.uniform(0, 360, 3), center="COM")
            mol.translate((np.array([i, j, k]) + site) * a - mol.get_center_of_mass())
            box += mol
    box.wrap()
    return box


def main():
    atoms = build()
    d = atoms.get_all_distances(mic=True)
    np.fill_diagonal(d, 9.0)
    print(f"{len(atoms)} atoms, box {BOX} A, min distance {d.min():.3f} A")
    comment = f"256 H2O on FCC sites of a 4x4x4 cell, random orientations (seed {SEED}), unrelaxed (build_box.py)"
    data = OUT.with_suffix(".data")
    write(data, atoms, format="lammps-data", atom_style="atomic", specorder=["H", "O"], masses=True)
    lines = data.read_text().splitlines()
    lines[0] = comment
    data.write_text("\n".join(lines) + "\n")
    atoms.info["comment"] = comment
    write(OUT.with_suffix(".extxyz"), atoms, format="extxyz")
    print(f"wrote {data} and {OUT.with_suffix('.extxyz')}")


if __name__ == "__main__":
    main()
