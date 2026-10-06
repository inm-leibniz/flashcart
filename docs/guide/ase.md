# ASE calculator

{class}`~flashcart.calculators.ase.FlashCartCalculator` evaluates energies,
forces, and stress with [ASE](https://docs.ase-lib.org). Provide a trained
checkpoint and a structure containing only elements supported by that model.
Reference labels are not required for prediction. The calculator does not
convert units. The examples below use models trained with energies in eV and
positions in Å.

```python
from ase.io import read
from flashcart.calculators import FlashCartCalculator

atoms = read("water_256.extxyz")
atoms.calc = FlashCartCalculator(checkpoint="models/my_run/best", skin=1.0)
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

## Calculator options

Checkpoints are loaded on CPU by default. Set `device="cuda"` to use a GPU.
The calculator accepts PyTorch device names. In training configurations,
`device: gpu` selects GPU execution.

- `add_atomic_offsets=True` restores the per-element energy shifts subtracted from the
  reference energies during training. It changes the total energy but not forces or
  stress. The default is `False`.
- `skin` adds a buffer to the neighbor-list radius without changing the model
  cutoff. For a fixed cell, the list can be reused until an atom moves by
  `skin / 2` from its position when the list was built. Changes to the cell
  or other atomic properties can also require rebuilding.
- `compile_mode="reduce-overhead"` enables `torch.compile` and graph padding. Padding
  keeps input shapes unchanged while the graph fits within the allocated capacities.
  Larger graphs increase those capacities and can require recompilation. Compilation
  adds an initial cost but can reduce the time required for repeated evaluations.

(ase-molecular-dynamics)=
## Molecular dynamics

The repository includes `examples/md/ase_md.py`, a Langevin molecular dynamics
script. From the repository root, provide a checkpoint directory and an initial
structure:

```bash
python examples/md/ase_md.py models/my_run/best structure.extxyz \
    --temperature 300 --timestep 0.5 --steps 2000
```

This runs 1 ps with a time step of 0.5 fs and a target temperature of 300 K.
The script uses the calculator's CPU default. Every 100 steps, it prints
potential, kinetic, and total energies per atom and writes a frame to `md.traj`.

Initial velocities are sampled at the requested temperature. The script does
not relax the structure before starting dynamics.

To use a GPU or compilation, adjust the calculator construction in the script using
the options above.

```{literalinclude} ../../examples/md/ase_md.py
:language: python
```
