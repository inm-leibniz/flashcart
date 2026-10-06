``flashcart.data``
==================

Represent atomic configurations and neighbor graphs, construct batches, and
compute the statistics used to initialize a model.
:class:`~flashcart.data.data.AtomicConfig` stores a configuration and its reference
properties as NumPy arrays. :class:`~flashcart.data.data.AtomicData` holds the
PyTorch tensors used for model evaluation, including atom types and neighbor
indices. Graph construction connects these representations using the model cutoff
and an ordered list of chemical elements.

See :doc:`../tutorials/datasets` for data preparation and reference-property keys.
The sections below describe the individual representations and processing steps.

``flashcart.data.data``
-----------------------

Represent atomic configurations and tensor graphs.

.. automodule:: flashcart.data.data

``flashcart.data.dataset``
--------------------------

Load datasets, build graph batches, and split data for training and evaluation.

For training, :func:`~flashcart.data.dataset.make_loader` combines dataset loading,
graph construction, and batching. Its ``batch_size`` argument is a target for one
rank. The training command divides the global target among ranks before calling
this function.

.. automodule:: flashcart.data.dataset

``flashcart.data.graph``
------------------------

Construct neighbor graphs from atomic configurations.

.. automodule:: flashcart.data.graph

``flashcart.data.neighbors``
----------------------------

Build and update neighbor lists.

.. automodule:: flashcart.data.neighbors

``flashcart.data.padding``
--------------------------

Pad graph batches for compiled evaluation using reusable tensor buffers.
Complete evaluation and backward passes before passing another batch to the
same padder.

.. automodule:: flashcart.data.padding
   :special-members: __call__

``flashcart.data.samplers``
---------------------------

Assign structures to ranks and group them into batches. Batches can be constrained
by atom and edge counts.

.. automodule:: flashcart.data.samplers

``flashcart.data.statistics``
-----------------------------

Compute dataset statistics used to initialize a potential.

.. automodule:: flashcart.data.statistics

``flashcart.data.utils``
------------------------

Convert atomic configurations and read or write structure data.

.. automodule:: flashcart.data.utils
