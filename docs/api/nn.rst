``flashcart.nn``
================

Build equivariant message-passing blocks from the operations in :doc:`o3`.
:class:`~flashcart.nn.layers.InteractionLayer` combines node features with edge
directions, applies distance-dependent weights, and sums the
messages at receiver atoms. :class:`~flashcart.nn.layers.ProductLayer` builds
higher-order correlations by repeatedly coupling the aggregated node features.

``flashcart.nn.layers``
-----------------------

Build interaction, product, gating, normalization, and readout layers.

In ``ProductLayer``, ``correlation`` controls the polynomial degree of the
recurrence, while the ``l_max`` parameters limit tensor rank. With path reduction,
paths are summed within each product. The contributions from all degrees are
added before a final linear map mixes channels. Otherwise, each recurrence step
mixes the retained paths and channels with a linear map. Gates, normalization,
and scalar readout layers complete the blocks used by
:class:`~flashcart.model.flashcart.FlashCartPotential`.

The ``LinearLayer`` in this module is a dense feature map. For an equivariant map
acting on tensor-rank blocks, use :class:`~flashcart.o3.linear.LinearLayer`.

.. automodule:: flashcart.nn.layers

``flashcart.nn.radial``
-----------------------

Compute radial basis expansions and cutoff functions.

.. automodule:: flashcart.nn.radial
