``flashcart.training``
======================

Train and evaluate potentials using configurable loss functions, optimizers, and
learning-rate schedules. :class:`~flashcart.training.tasks.TrainingTask` runs
training, validation, and checkpointing, while
:class:`~flashcart.training.tasks.EvaluationTask` evaluates a model over a loader
and combines batch statistics into the requested metrics.

Loss objects specify both the target property and the predictions needed to
compute it. The task uses these requirements to request energies, forces, or
stress and to retain the derivative graph when needed for training. Optimizer
and scheduler functions read the configuration to create the update rules.

See :doc:`../quickstart` for the training
workflow and :doc:`../guide/multi_gpu` for distributed execution.

``flashcart.training.tasks``
----------------------------

Run training, validation, checkpointing, and model evaluation.

The ``best`` checkpoint is selected by the validation metric. The ``log``
checkpoint stores the state needed to resume training. Despite the argument name
``early_stopping_loss``, training continues to ``max_epochs`` even if the
validation metric stops improving.

.. automodule:: flashcart.training.tasks

``flashcart.training.loss``
---------------------------

Define property loss functions and combine their batch statistics.

.. automodule:: flashcart.training.loss
   :special-members: __call__

``flashcart.training.optimizers``
---------------------------------

Construct optimizers from model parameters and configuration settings.

.. automodule:: flashcart.training.optimizers

``flashcart.training.schedulers``
---------------------------------

Construct learning-rate schedules with warmup and decay.

.. automodule:: flashcart.training.schedulers

``flashcart.training.muon``
---------------------------

.. automodule:: flashcart.training.muon

``flashcart.training.muon_norm``
--------------------------------

.. automodule:: flashcart.training.muon_norm
