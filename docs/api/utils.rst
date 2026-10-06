``flashcart.utils``
===================

These functions handle configuration, distributed execution, graph geometry,
compilation, logging, and reductions. They support the interfaces described
in :doc:`model`, :doc:`calculators`, and :doc:`cli`.

Use the configuration helpers to combine defaults, YAML files, and overrides in
a Python script. The geometry helpers construct edge vectors and
convert stress tensors to the Voigt convention used by model predictions.
Compilation settings affect the PyTorch process. The distributed helpers manage
execution on individual ranks, and the LAMMPS helpers exchange atomic features
between ranks.

For user-facing settings and examples, see :doc:`../guide/configuration`,
:doc:`../guide/env_vars`, and :doc:`../guide/multi_gpu`.

``flashcart.utils.config``
--------------------------

Read configuration files and combine defaults with supplied settings.

.. automodule:: flashcart.utils.config

``flashcart.utils.distributed``
-------------------------------

Manage batch sizes and loader iteration across distributed ranks.

.. automodule:: flashcart.utils.distributed

``flashcart.utils.compile``
---------------------------

Configure PyTorch compilation for model evaluation.

.. automodule:: flashcart.utils.compile

``flashcart.utils.env``
-----------------------

Read Boolean and integer settings from environment variables.

.. automodule:: flashcart.utils.env

``flashcart.utils.geometry``
----------------------------

Compute edge vectors and convert stress tensors.

.. automodule:: flashcart.utils.geometry

``flashcart.utils.lammps``
--------------------------

Exchange atomic features and their gradients between LAMMPS ranks. Forward and
reverse exchanges are PyTorch custom operators that support compiled predictions.
When calling the exchange helpers directly, keep both evaluation and
differentiation inside :func:`~flashcart.utils.lammps.lammps_data_slot`, including
on ranks without ghost atoms. The LAMMPS calculator manages this context
automatically.

.. autofunction:: flashcart.utils.lammps.lammps_exchange_features

.. autofunction:: flashcart.utils.lammps.lammps_data_slot

.. autofunction:: flashcart.utils.lammps.lammps_forward_exchange

.. autofunction:: flashcart.utils.lammps.lammps_reverse_exchange

``flashcart.utils.logging``
---------------------------

Configure log output and report training progress.

.. automodule:: flashcart.utils.logging

``flashcart.utils.parameter_groups``
------------------------------------

Collect parameters assigned to specific optimizer groups.

.. automodule:: flashcart.utils.parameter_groups

``flashcart.utils.scatter``
---------------------------

Sum tensor values by index.

.. automodule:: flashcart.utils.scatter
