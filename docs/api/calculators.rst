``flashcart.calculators``
=========================

Use trained potentials in atomistic simulations. The ASE calculator constructs
neighbor graphs from ``ase.Atoms`` objects and returns predictions using ASE's
calculator interface. The LAMMPS interface receives neighbor and atom data
from ML-IAP and exchanges features between owned and ghost atoms when needed.

Start with :doc:`../guide/ase` for Python simulations or
:doc:`../guide/lammps` for export, build requirements, and simulation
input. The reference below describes the calculator options and the lower-level
ML-IAP graph and export helpers.

``flashcart.calculators.ase``
-----------------------------

Evaluate trained potentials with ASE.

.. automodule:: flashcart.calculators.ase

``flashcart.calculators.lammps_mliap``
--------------------------------------

Build ML-IAP graphs, export models, and evaluate potentials in LAMMPS.

.. automodule:: flashcart.calculators.lammps_mliap
   :exclude-members: compute_descriptors, compute_gradients
