"""Run Langevin molecular dynamics with ASE and a trained FlashCart model.

    python ase_md.py models/i3c3f64l2_s0/best water/water_256.extxyz --steps 2000

Prints the potential, kinetic, and total energy every 100 steps and writes
every 100th frame to md.traj (view it with ``ase gui md.traj``). The water
box comes from water/build_box.py.
"""

import argparse

from ase import units
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from flashcart.calculators import FlashCartCalculator


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", help="checkpoint directory, e.g. <output_path>/best")
    parser.add_argument("structure", help="any file ASE can read")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=300.0, help="K")
    parser.add_argument("--timestep", type=float, default=0.5, help="fs")
    args = parser.parse_args()

    atoms = read(args.structure)
    atoms.calc = FlashCartCalculator(checkpoint=args.checkpoint, skin=1.0)
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature)
    dyn = Langevin(
        atoms, args.timestep * units.fs, temperature_K=args.temperature, friction=0.01 / units.fs, fixcm=False
    )
    traj = Trajectory("md.traj", "w", atoms)

    def report():
        epot, ekin = atoms.get_potential_energy(), atoms.get_kinetic_energy()
        n = len(atoms)
        print(f"step {dyn.nsteps:6d}  Epot {epot / n:9.4f}  Ekin {ekin / n:7.4f}  Etot {(epot + ekin) / n:9.4f} eV/atom")

    dyn.attach(report, interval=100)
    dyn.attach(traj.write, interval=100)
    dyn.run(args.steps)


if __name__ == "__main__":
    main()
