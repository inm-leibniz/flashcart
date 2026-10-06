``flashcart.cli``
=================

The console scripts call the entry points below. Use ``flashcart-train`` to
train a potential, ``flashcart-test`` to evaluate a saved checkpoint, and
``flashcart-lammps`` to export a model for the ML-IAP interface.
``flashcart-profile`` measures model evaluation and differentiation on dataset
batches. It constructs a model from the profiling configuration and does not
perform optimizer updates.

Start with :doc:`../quickstart` for command examples. See
:ref:`configuration-overrides` for configuration files and ``KEY=VALUE`` overrides,
:ref:`profiling-command` for profiling, and :ref:`lammps-export` for model export.

Call :func:`~flashcart.cli.train.train`, :func:`~flashcart.cli.test.test`, and
:func:`~flashcart.cli.profile.profile` from Python with configuration overrides.
The ``main`` functions parse command-line arguments. For model export from Python,
use the helpers in :doc:`calculators`.

``flashcart.cli.train``
-----------------------

.. automodule:: flashcart.cli.train

``flashcart.cli.test``
----------------------

.. automodule:: flashcart.cli.test

``flashcart.cli.profile``
-------------------------

.. automodule:: flashcart.cli.profile

``flashcart.cli.lammps_mliap``
------------------------------

.. automodule:: flashcart.cli.lammps_mliap
