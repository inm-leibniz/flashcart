``flashcart.model``
===================

Construct, save, and load interatomic potentials, and evaluate their properties.
:class:`~flashcart.model.flashcart.FlashCartPotential` implements the energy model.
:class:`~flashcart.model.atomistic.AtomisticModel` provides the prediction and
checkpoint interfaces shared by atomistic models.

Call :meth:`~flashcart.model.flashcart.FlashCartPotential.forward` for site energies
or :meth:`~flashcart.model.atomistic.AtomisticModel.predict` for total energies and
requested derivatives. Computing energy derivatives requires PyTorch gradient tracking,
even during model evaluation. When training on these derivatives,
``create_graph=True`` keeps them differentiable with respect to model parameters.
Direct model predictions omit the stored per-element energy shifts. Calculators
can restore these shifts when reporting energies.

For evaluation from ASE structures, start with :doc:`calculators` and
:doc:`../guide/ase`. The interfaces below are useful when constructing
models or supplying atomic graphs directly.

For LAMMPS, use the interface described in :doc:`../guide/lammps`. When supplying
LAMMPS graph dictionaries directly, set ``lammps_exchange=True``, provide edge vectors,
and place owned atoms before ghost atoms. ``batch`` covers only owned atoms, while
``atom_types`` covers both. The model infers the counts from these tensor shapes
and returns site energies only for owned atoms. With more than one interaction
layer, keep evaluation and differentiation inside
:func:`~flashcart.utils.lammps.lammps_data_slot`, including on ranks without ghost
atoms. The LAMMPS calculator manages this context automatically.

``flashcart.model.flashcart``
-----------------------------

Construct a FlashCart potential from equivariant interaction and product layers.

``FlashCartPotential`` inherits its prediction and checkpoint interfaces from
``AtomisticModel``. Use :meth:`~flashcart.model.atomistic.AtomisticModel.predict`
to evaluate a graph, :meth:`~flashcart.model.atomistic.AtomisticModel.from_checkpoint`
to load a saved potential, and
:meth:`~flashcart.model.atomistic.AtomisticModel.save_inference_checkpoint` to save
its configuration and weights. The separate
:meth:`~flashcart.model.atomistic.AtomisticModel.load_weights` and
:meth:`~flashcart.model.atomistic.AtomisticModel.save_weights` methods operate on
weights alone.

.. automodule:: flashcart.model.flashcart

``flashcart.model.atomistic``
-----------------------------

Predict atomic properties and save or load model checkpoints.

.. automodule:: flashcart.model.atomistic
