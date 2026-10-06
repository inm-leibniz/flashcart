import logging
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, NoReturn, Optional, Union

# Neighbor counts change during molecular dynamics, changing the size of temporary
# tensors used to compute edge forces. Expandable segments limit fragmentation in
# PyTorch's CUDA memory cache as these sizes change.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import ase.data
import numpy as np
import torch

from flashcart.utils.env import env_bool, env_int
from flashcart.model.flashcart import FlashCartPotential
from flashcart.utils.lammps import lammps_data_slot
from flashcart.utils.torch_geometric.data import Data

COMPILE_MODES = ("default", "max-autotune-no-cudagraphs")

try:
    from lammps.mliap.mliap_unified_abc import MLIAPUnified
except ImportError:

    class MLIAPUnified:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass


def graph_from_mliap(
    data,
    element_types: List[str],
    r_max: float,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    cell: Optional[np.ndarray] = None,
    pbc: Optional[np.ndarray] = None,
    validate: bool = False,
) -> Data:
    """Build a FlashCart graph from a LAMMPS ML-IAP data object.

    LAMMPS supplies edge vectors through ``rij``, including periodic translations. The
    graph stores these vectors directly and sets ``use_shifts=False``, so positions and
    cell data are not used to reconstruct them. Nodes include both owned and ghost
    atoms, while ``batch`` and ``n_atoms`` describe only the ``nlocal`` owned atoms.
    Site energies are therefore summed over owned atoms. ``lammps_exchange=True``
    selects this layout in the model. When ghost features must be exchanged between
    layers, evaluation and differentiation require an active
    :func:`~flashcart.utils.lammps.lammps_data_slot` context.

    Args:
        data: LAMMPS ML-IAP data object (``ntotal``, ``nlocal``, ``npairs``, ``pair_i``,
            ``pair_j``, ``rij``, ``elems``).
        element_types (list[str]): Chemical symbols in ML-IAP element order, indexed
            by ``data.elems``.
        r_max (float): Model cutoff radius, stored as graph metadata. The neighbor list
            provided by LAMMPS is not rebuilt or filtered.
        device (torch.device, optional): Requested tensor device. CUDA input arrays
            require a CUDA device. Default: inferred from the input arrays.
        dtype (torch.dtype, optional): Floating-point dtype. Default:
            ``torch.get_default_dtype()``.
        cell (np.ndarray, optional): Cell matrix stored as graph metadata. Default: a
            diagonal matrix with entries of 1000.
        pbc (np.ndarray, optional): Periodic boundary flags stored as graph metadata.
            Default: all False.
        validate (bool, optional): Check that atom and element-type indices lie within
            their allowed ranges. Default: False.

    Returns:
        Data: Graph containing owned and ghost atoms and the edge vectors provided by
            LAMMPS. ``n_atoms`` and ``batch`` cover only owned atoms.
    """
    ntotal, nlocal, nghost, npairs, device, dtype, pair_i, pair_j, vectors, atom_types = _mliap_core_tensors(
        data,
        element_types,
        device=device,
        dtype=dtype,
        validate=validate,
    )
    z_table = torch.tensor([ase.data.atomic_numbers[s] for s in element_types], dtype=torch.long, device=device)

    if cell is None:
        cell_t = torch.eye(3, dtype=dtype, device=device).unsqueeze(0) * 1.0e3
    else:
        cell_t = torch.as_tensor(cell, dtype=dtype, device=device)
        if cell_t.shape == (3, 3):
            cell_t = cell_t.unsqueeze(0)
    if pbc is None:
        pbc_t = torch.zeros(1, 3, dtype=torch.bool, device=device)
    else:
        pbc_t = torch.as_tensor(pbc, dtype=torch.bool, device=device)
        if pbc_t.ndim == 0:
            pbc_t = pbc_t.expand(3)
        if pbc_t.shape == (3,):
            pbc_t = pbc_t.unsqueeze(0)

    graph = Data(
        num_nodes=ntotal,
        n_atoms=torch.tensor([nlocal], dtype=torch.long, device=device),
        atomic_numbers=z_table[atom_types],
        positions=torch.zeros(ntotal, 3, dtype=dtype, device=device),
        cell=cell_t,
        pbc=pbc_t,
        edge_index=torch.stack([pair_i, pair_j], dim=0),
        shifts=torch.zeros(npairs, 3, dtype=torch.long, device=device),
        atom_types=atom_types,
        vectors=vectors,
        batch=torch.zeros(nlocal, dtype=torch.long, device=device),
        use_shifts=False,
        lammps_exchange=True,
        nlocal=nlocal,
        ntotal=ntotal,
        nghost=nghost,
    )
    graph._r_max = r_max
    graph._skin = 0.0
    return graph


def graph_dict_from_mliap(
    data,
    element_types: List[str],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    validate: bool = False,
) -> Dict[str, object]:
    """Build a graph dictionary for energy and edge-force evaluation.

    The dictionary uses the same atom and edge representation as ``graph_from_mliap``
    but omits positions, cell data, and periodic shifts. LAMMPS supplies the edge
    vectors directly, and edge forces are obtained by differentiating the energy with
    respect to these vectors. The dictionary contains only tensors and flags. With
    ``lammps_exchange=True``, the model infers owned and total atom counts from the
    shapes of ``batch`` and ``atom_types`` for dynamic compilation. When ghost features
    must be exchanged between layers, evaluation and differentiation require an active
    :func:`~flashcart.utils.lammps.lammps_data_slot` context.

    Args:
        data: LAMMPS ML-IAP data object, as in ``graph_from_mliap``.
        element_types (list[str]): Chemical symbols in ML-IAP element order, indexed
            by ``data.elems``.
        device (torch.device, optional): Requested tensor device. CUDA input arrays
            require a CUDA device. Default: inferred from the input arrays.
        dtype (torch.dtype, optional): Floating-point dtype. Default:
            ``torch.get_default_dtype()``.
        validate (bool, optional): Check that atom and element-type indices lie within
            their allowed ranges. Default: False.

    Returns:
        dict[str, object]: Graph dictionary containing atom types, connectivity, and
            edge vectors, without positions, cell data, or periodic shifts.
    """
    _, nlocal, _, _, _, _, pair_i, pair_j, vectors, atom_types = _mliap_core_tensors(
        data,
        element_types,
        device=device,
        dtype=dtype,
        validate=validate,
    )
    z_table = torch.tensor([ase.data.atomic_numbers[s] for s in element_types], dtype=torch.long, device=vectors.device)
    graph_dict: Dict[str, object] = {
        "n_atoms": torch.tensor([nlocal], dtype=torch.long, device=vectors.device),
        "edge_index": torch.stack([pair_i, pair_j], dim=0),
        "atomic_numbers": z_table[atom_types],
        "atom_types": atom_types,
        "vectors": vectors,
        "batch": torch.zeros(nlocal, dtype=torch.long, device=vectors.device),
        "use_shifts": False,
        "lammps_exchange": True,
    }
    return graph_dict


def _mliap_core_tensors(
    data,
    element_types: List[str],
    device: Optional[torch.device],
    dtype: Optional[torch.dtype],
    validate: bool,
):
    """Convert LAMMPS atom and neighbor-pair arrays to graph tensors.

    The arrays are restricted to the reported atom and pair counts. Pair indices and
    element types are converted to integer tensors, while edge vectors use the requested
    floating-point dtype. Optional validation checks that atom and element-type indices
    lie within their allowed ranges.

    Args:
        data: LAMMPS ML-IAP data object containing atom counts, pair indices, edge
            vectors, and element types.
        element_types (list[str]): Chemical symbols in ML-IAP element order, indexed
            by ``data.elems``.
        device (torch.device or None): Requested tensor device. If None, infer the
            device from the input arrays. CUDA input arrays require a CUDA device.
        dtype (torch.dtype or None): Floating-point dtype for edge vectors. If None, use
            ``torch.get_default_dtype()``.
        validate (bool): Whether to check atom and element-type indices.

    Returns:
        tuple: Values in the order ``ntotal``, ``nlocal``, ``nghost``, ``npairs``,
            ``device``, ``dtype``, ``pair_i``, ``pair_j``, ``vectors``, and
            ``atom_types``. The first four values are integer counts, followed by the
            resolved device and dtype. Pair-index tensors have shape ``(npairs,)``, edge
            vectors have shape ``(npairs, 3)``, and element-type indices have shape
            ``(ntotal,)``.
    """
    ntotal = int(data.ntotal)
    nlocal = int(data.nlocal)
    nghost = int(getattr(data, "nghosts", getattr(data, "nghost", ntotal - nlocal)))
    npairs = int(data.npairs)
    if npairs <= 0 or ntotal <= 0:
        raise ValueError(f"Empty ML-IAP layout: ntotal={ntotal}, npairs={npairs}.")

    dtype = torch.get_default_dtype() if dtype is None else dtype
    pair_i_arr = _mliap_array(data, "pair_i")
    pair_j_arr = _mliap_array(data, "pair_j")
    vectors_arr = _mliap_array(data, "rij")
    elems_arr = _mliap_array(data, "elems")
    device = _array_device((pair_i_arr, pair_j_arr, vectors_arr, elems_arr), device)

    pair_i = torch.as_tensor(pair_i_arr, dtype=torch.long, device=device)[:npairs]
    pair_j = torch.as_tensor(pair_j_arr, dtype=torch.long, device=device)[:npairs]
    vectors = torch.as_tensor(vectors_arr, dtype=dtype, device=device)[:npairs]
    elems = torch.as_tensor(elems_arr, dtype=torch.long, device=device)[:ntotal]
    if validate:
        _validate_lammps_indices("pair_i", pair_i, ntotal)
        _validate_lammps_indices("pair_j", pair_j, ntotal)
    if vectors.ndim != 2 or vectors.shape[-1] != 3:
        raise ValueError(f"LAMMPS rij must have shape (npairs, 3), got {tuple(vectors.shape)}.")

    atom_types = elems
    if validate:
        _validate_lammps_indices("atom_types", atom_types, len(element_types))
    if validate or env_bool("FLASHCART_LAMMPS_DEBUG"):
        _log_lammps_graph_debug(ntotal, nlocal, nghost, npairs, elems, atom_types, pair_i, pair_j, device)
    return ntotal, nlocal, nghost, npairs, device, dtype, pair_i, pair_j, vectors, atom_types


def _mliap_array(data, name: str):
    """Read an ML-IAP array and explain missing CuPy support for CUDA arrays.

    Args:
        data: ML-IAP object containing the requested array.
        name (str): Array attribute to read.

    Returns:
        Any: Array exposed by the ML-IAP attribute.

    Raises:
        RuntimeError: Accessing a CUDA array fails because CuPy is unavailable.
    """
    try:
        return getattr(data, name)
    except NameError as exc:
        if "cupy" in str(exc):
            raise RuntimeError(
                "LAMMPS Kokkos ML-IAP provided CUDA arrays, but CuPy is not available. "
                "Install a matching wheel, for example: pip install cupy-cuda12x."
            ) from exc
        raise


def _array_device(values, requested: Optional[torch.device]) -> torch.device:
    """Select a tensor device compatible with the ML-IAP input arrays.

    Arrays exposing ``__cuda_array_interface__`` require CUDA. For these arrays, retain
    a requested CUDA device. Otherwise, select the default CUDA device if CUDA is
    available and CPU execution is not forced. For other arrays, use the requested
    device or CPU if none is specified.

    Args:
        values: Input arrays whose device interfaces are inspected.
        requested (torch.device or None): Requested tensor device.

    Returns:
        torch.device: Device selected for tensor conversion.

    Raises:
        RuntimeError: CUDA input arrays require automatic CUDA selection, but CUDA is
            unavailable or ``FLASHCART_LAMMPS_CPU`` is enabled.
    """
    if any(hasattr(value, "__cuda_array_interface__") for value in values):
        requested = None if requested is None else torch.device(requested)
        if requested is not None and requested.type == "cuda":
            return requested
        if not torch.cuda.is_available() or env_bool("FLASHCART_LAMMPS_CPU"):
            raise RuntimeError(
                "LAMMPS provides CUDA ML-IAP arrays, but FlashCart is configured for CPU. "
                "Expose CUDA to this process, or set up LAMMPS to pass CPU ML-IAP arrays."
            )
        return torch.device("cuda")
    return torch.device("cpu") if requested is None else torch.device(requested)


def _tensor_range(values: torch.Tensor) -> tuple[int, int]:
    """Return integer minimum and maximum values, or ``(0, -1)`` for an empty tensor."""
    if values.numel() == 0:
        return 0, -1
    return int(values.min().item()), int(values.max().item())


def _validate_lammps_indices(name: str, indices: torch.Tensor, size: int) -> None:
    """Check that every index lies in the half-open interval ``[0, size)``.

    Args:
        name (str): Array name included in an error message.
        indices (torch.Tensor): Integer indices to validate.
        size (int): Exclusive upper bound.

    Raises:
        ValueError: At least one index is negative or greater than or equal to size.
    """
    lo, hi = _tensor_range(indices)
    if lo < 0 or hi >= size:
        raise ValueError(f"LAMMPS {name} contains indices in [{lo}, {hi}], expected [0, {size}).")


def _log_lammps_graph_debug(
    ntotal: int,
    nlocal: int,
    nghost: int,
    npairs: int,
    elems: torch.Tensor,
    atom_types: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    device: torch.device,
) -> None:
    """Log atom counts and index ranges when ``FLASHCART_LAMMPS_DEBUG`` is enabled.

    Args:
        ntotal (int): Number of owned and ghost atoms.
        nlocal (int): Number of owned atoms.
        nghost (int): Number of ghost atoms.
        npairs (int): Number of neighbor pairs.
        elems (torch.Tensor): Element indices received from ML-IAP.
        atom_types (torch.Tensor): Element indices used by the model graph.
        pair_i (torch.Tensor): Receiver indices.
        pair_j (torch.Tensor): Sender indices.
        device (torch.device): Device holding the graph tensors.
    """
    if not env_bool("FLASHCART_LAMMPS_DEBUG"):
        return
    elem_lo, elem_hi = _tensor_range(elems)
    atom_lo, atom_hi = _tensor_range(atom_types)
    i_lo, i_hi = _tensor_range(pair_i)
    j_lo, j_hi = _tensor_range(pair_j)
    msg = (
        "FlashCart LAMMPS graph: ntotal=%d nlocal=%d nghost=%d npairs=%d device=%s "
        "elems=[%d,%d] atom_types=[%d,%d] pair_i=[%d,%d] pair_j=[%d,%d]"
    ) % (
        ntotal,
        nlocal,
        nghost,
        npairs,
        device,
        elem_lo,
        elem_hi,
        atom_lo,
        atom_hi,
        i_lo,
        i_hi,
        j_lo,
        j_hi,
    )
    logging.info(msg)
    print(msg, flush=True)


def _log_cuda_memory(label: str, device: torch.device) -> None:
    """Print and log CUDA memory use when ``FLASHCART_LAMMPS_MEMORY_DEBUG`` is enabled.

    Args:
        label (str): Evaluation stage included in the message.
        device (torch.device): CUDA device to inspect. Other device types are ignored.
    """
    if not env_bool("FLASHCART_LAMMPS_MEMORY_DEBUG"):
        return
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    msg = (
        "FlashCart CUDA memory %s: allocated=%.3f GB reserved=%.3f GB "
        "max_allocated=%.3f GB max_reserved=%.3f GB device_used=%.3f GB"
    ) % (
        label,
        torch.cuda.memory_allocated(device) / 1024**3,
        torch.cuda.memory_reserved(device) / 1024**3,
        torch.cuda.max_memory_allocated(device) / 1024**3,
        torch.cuda.max_memory_reserved(device) / 1024**3,
        (total_bytes - free_bytes) / 1024**3,
    )
    logging.info(msg)
    print(msg, flush=True)


def _maybe_empty_cuda_cache(device: torch.device, step: int) -> None:
    """Release unused CUDA cache blocks at the configured force-call interval.

    The operation requires ``FLASHCART_LAMMPS_EMPTY_CACHE``. The interval comes from
    ``FLASHCART_LAMMPS_EMPTY_CACHE_EVERY`` and is at least one call.

    Args:
        device (torch.device): CUDA device used by the model.
        step (int): Number of evaluated force calls so far.
    """
    if not env_bool("FLASHCART_LAMMPS_EMPTY_CACHE"):
        return
    every = max(env_int("FLASHCART_LAMMPS_EMPTY_CACHE_EVERY", 1), 1)
    if step % every != 0:
        return
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    releasable = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    if releasable <= 0:
        return
    torch.cuda.empty_cache()
    _log_cuda_memory("after empty_cache", device)


class _IdentityExchange:
    """Copy features and gradients when ML-IAP provides no exchange methods.

    Used only when no ghost atoms are present. When ML-IAP provides exchange
    methods, the calculator uses them even on ranks without ghost atoms.
    """

    def forward_exchange(self, feats, out, vec_len: int) -> None:
        """Copy owned-atom features to the output buffer.

        Args:
            feats (torch.Tensor): Features with shape ``(nlocal, vec_len)``.
            out (torch.Tensor): Output buffer with the same shape as ``feats``.
            vec_len (int): Number of features per atom. Unused by the copy operation.
        """
        out.copy_(feats)

    def reverse_exchange(self, grads, out, vec_len: int) -> None:
        """Copy owned-atom gradients to the output buffer.

        Args:
            grads (torch.Tensor): Gradients with shape ``(nlocal, vec_len)``.
            out (torch.Tensor): Output buffer with the same shape as ``grads``.
            vec_len (int): Number of features per atom. Unused by the copy operation.
        """
        out.copy_(grads)


def _exchange_object(data, nlocal: int, ntotal: int):
    """Return the object that performs ghost-feature exchange for this force call.

    Kokkos ML-IAP provides ``forward_exchange`` and ``reverse_exchange``. Without
    these methods, models with multiple interaction layers are supported only when
    no ghost atoms are present. In that case, the returned object copies local
    features and gradients.

    Args:
        data: ML-IAP data object for the current force evaluation.
        nlocal (int): Number of owned atoms.
        ntotal (int): Number of owned and ghost atoms.

    Returns:
        object: The data object when it provides both exchange hooks, or an
            ``_IdentityExchange`` instance when no ghost atoms are present.

    Raises:
        RuntimeError: Ghost atoms are present but the data object has no exchange hooks.
    """
    if hasattr(data, "forward_exchange") and hasattr(data, "reverse_exchange"):
        return data
    if ntotal == nlocal:
        return _IdentityExchange()
    raise RuntimeError(
        "Ghost atoms require the ML-IAP communication hooks, which only the KOKKOS ML-IAP build provides. Build LAMMPS with PKG_KOKKOS and run with `-k on ... -sf kk`, or use a model with one interaction layer."
    )


class FlashCartLAMMPSUnified(MLIAPUnified):
    """Evaluate a FlashCart potential through the LAMMPS ML-IAP interface.

    ``compute_forces`` evaluates site energies for owned atoms and obtains edge forces
    by differentiating the energy with respect to the edge vectors provided by LAMMPS.
    Models with multiple interaction layers exchange ghost-atom features between layers
    through ML-IAP, with reverse communication during differentiation. Every rank
    participates in feature exchange, including ranks without ghost atoms, because
    other ranks may need features from its atoms. Ranks without owned atoms or
    neighbor pairs are not supported.

    Construction moves the provided model to CPU, sets evaluation mode, and disables
    gradients for its parameters. During evaluation, the model moves to the device
    selected for the ML-IAP arrays.

    Setting ``compile_mode`` enables compilation with ``torch.compile`` and dynamic
    shapes. Compilation starts on the first model evaluation, after the model has
    moved to the device used by the ML-IAP arrays. Dynamic shapes allow graph reuse
    as atom and neighbor-pair counts change, although some changes can still require
    recompilation. The compiled function is not saved with
    the interface. Each LAMMPS process creates its own compiled function. The PyTorch
    Inductor cache (``TORCHINDUCTOR_CACHE_DIR``) can shorten later compilations.

    Args:
        model (FlashCartPotential): Potential to evaluate.
        add_atomic_offsets (bool, optional): Restore the per-element energy shifts
            subtracted from the reference energies during training. These shifts affect
            the reported energies but not the forces. Default: False.
        compile_mode (str, optional): ``torch.compile`` mode: ``"default"`` or
            ``"max-autotune-no-cudagraphs"``. Modes using CUDA graphs are not supported
            by this interface. Default: None, which evaluates the model without
            compilation.
        compile_fullgraph (bool, optional): Whether compilation requires a single graph
            without graph breaks. Used when ``compile_mode`` is set. Default: True.

    Raises:
        ValueError: ``compile_mode`` is not supported.
    """

    def __init__(
        self,
        model: FlashCartPotential,
        add_atomic_offsets: bool = False,
        compile_mode: Optional[str] = None,
        compile_fullgraph: bool = True,
    ):
        super().__init__()
        if compile_mode is not None and compile_mode not in COMPILE_MODES:
            raise ValueError(f"compile_mode must be one of {COMPILE_MODES} or None, got {compile_mode!r}.")
        self.model = model
        self.element_types = list(model.elements)
        self.r_max = float(model.r_max)
        self.rcutfac = 0.5 * self.r_max
        self.ndescriptors = 1
        self.nparams = 1
        self.dtype = next(model.parameters()).dtype
        self.add_atomic_offsets = bool(add_atomic_offsets)
        self.compile_mode = compile_mode
        self.compile_fullgraph = bool(compile_fullgraph)
        self._atomic_shifts = model.scale_shift.shifts.detach().to(torch.float64)

        self.device = torch.device("cpu")
        self.model = self.model.to(self.device).eval()
        self._initialized = False
        self._checked_lammps_graph = False
        self._step = 0
        self._freeze_model_parameters()

    def _predict_fn(self) -> Callable[[Dict[str, object]], Dict[str, torch.Tensor]]:
        """Return a function that predicts energies and edge forces on the current device.

        Compiled functions are cached on the model by prediction and compilation
        options. Compilation occurs when the returned function is first called.
        Moving the model to another device clears its cache.

        Returns:
            Callable: Function accepting a graph dictionary with differentiable edge
                vectors and returning site energies, total energy, and edge forces.
        """
        if self.compile_mode is None:
            return lambda graph: self.model._predict_impl(
                graph,
                compute_forces=False,
                compute_stress=False,
                compute_edge_forces=True,
                create_graph=False,
            )
        return self.model._compiled_predict_impl(
            compute_forces=False,
            compute_stress=False,
            compute_edge_forces=True,
            create_graph=False,
            compile_mode=self.compile_mode,
            fullgraph=self.compile_fullgraph,
            dynamic=True,
        )

    def compute_descriptors(self, data) -> None:
        """Satisfy the ML-IAP descriptor hook without computing separate descriptors.

        Args:
            data: ML-IAP data object. Unused because ``compute_forces`` evaluates
                the model.
        """
        pass

    def compute_gradients(self, data) -> None:
        """Satisfy the ML-IAP gradient hook without computing descriptor gradients.

        Args:
            data: ML-IAP data object. Unused because ``compute_forces`` evaluates
                forces.
        """
        pass

    def _freeze_model_parameters(self) -> None:
        """Disable parameter gradients while retaining input gradients for forces."""
        for param in self.model.parameters():
            param.requires_grad_(False)

    def _initialize(self, data) -> None:
        """Freeze model parameters and mark the interface as initialized.

        Args:
            data: ML-IAP data object supplied by the callback. Not inspected here.
        """
        self._freeze_model_parameters()
        self._initialized = True
        logging.info(
            "FlashCart LAMMPS: model on %s, r_max=%.4f, compile_mode=%s", self.device, self.r_max, self.compile_mode
        )

    def _fail(self, message: str) -> NoReturn:
        """Report an unsupported configuration and terminate an attached LAMMPS process.

        A callback exception enters a LAMMPS barrier that other ranks may never reach.
        Exiting immediately with a nonzero status lets the MPI launcher terminate the
        job instead. The process exits immediately without running Python cleanup
        handlers. Calls without an attached LAMMPS interface raise ``RuntimeError``.

        Args:
            message (str): Explanation of the unsupported configuration.

        Raises:
            RuntimeError: No LAMMPS interface is attached.
        """
        if getattr(self, "interface", None) is not None:
            try:
                print(f"FlashCart LAMMPS: {message}", file=sys.stderr, flush=True)
            finally:
                os._exit(1)
        raise RuntimeError(message)

    def compute_forces(self, data) -> None:
        """Evaluate forces and write the requested energies to the ML-IAP data.

        Edge forces are passed to ``data.update_pair_forces`` when available. Otherwise,
        their contributions are accumulated for owned and ghost atoms and written to
        ``data.f``. LAMMPS returns ghost-atom contributions to their owners through
        reverse communication.

        When ``data.eflag`` is true or absent, the method writes the summed energy of
        the owned atoms and their site energies when an ``eatoms`` buffer is available.
        Each rank must own at least one atom and have at least one neighbor pair.
        With several interaction layers, the ghost-feature exchange is matched between
        neighboring ranks, and a rank that cannot evaluate the model would leave them
        waiting. Unsupported configurations terminate an attached LAMMPS process with
        a nonzero status. Standalone calls raise ``RuntimeError``.

        Atom and element-type indices are checked for the first evaluated graph. Setting
        ``FLASHCART_LAMMPS_VALIDATE=1`` or ``FLASHCART_LAMMPS_DEBUG=1`` enables these
        checks for every evaluated graph.

        Args:
            data: LAMMPS ML-IAP data object for the current step.
        """
        nlocal = int(data.nlocal)
        ntotal = int(data.ntotal)
        npairs = int(data.npairs)
        if nlocal == 0 or npairs <= 0:
            self._fail(
                f"This rank owns {nlocal} atoms with {npairs} neighbor pairs. Every rank must own atoms with "
                "neighbor pairs. For empty regions, try the LAMMPS `balance` command or fewer ranks. "
                "Configurations with no neighbor pairs remain unsupported on one rank."
            )
        self._step += 1

        _log_cuda_memory("before graph", self.device)
        checked_lammps_graph = getattr(self, "_checked_lammps_graph", False)
        validate_graph = (
            not checked_lammps_graph or env_bool("FLASHCART_LAMMPS_VALIDATE") or env_bool("FLASHCART_LAMMPS_DEBUG")
        )
        graph = graph_dict_from_mliap(
            data,
            self.element_types,
            device=self.device,
            dtype=self.dtype,
            validate=validate_graph,
        )
        self._checked_lammps_graph = True
        graph_device = graph["vectors"].device
        if graph_device != self.device:
            self.device = graph_device
            self.model = self.model.to(self.device)
        if not self._initialized:
            self._initialize(data)
        _log_cuda_memory("after graph", self.device)

        graph["vectors"] = graph["vectors"].detach().requires_grad_(True)
        exchange_data = data
        if self.model.n_interactions > 1:
            try:
                exchange_data = _exchange_object(data, nlocal, ntotal)
            except RuntimeError as exc:
                self._fail(str(exc))
        with lammps_data_slot(exchange_data):
            out = self._predict_fn()(graph)
        _log_cuda_memory("after predict", self.device)

        pair_forces = out["edge_forces"].detach()
        if pair_forces.dtype != torch.float64:
            pair_forces = pair_forces.to(torch.float64)

        if bool(getattr(data, "eflag", True)):
            node_energies = out["node_energies"].detach()
            if node_energies.dtype != torch.float64:
                node_energies = node_energies.to(torch.float64)
            e_local = node_energies[:nlocal]
            if self.add_atomic_offsets:
                atom_types_local = graph["atom_types"][:nlocal]
                e_local = e_local + self._atomic_shifts.to(e_local.device)[atom_types_local]
            if _has_mliap_attr(data, "eatoms"):
                try:
                    eatoms_buf = _mliap_array(data, "eatoms")
                except AttributeError:
                    data.eatoms = e_local.cpu().numpy()
                else:
                    if eatoms_buf is not None:
                        eatoms_tensor = torch.as_tensor(eatoms_buf)
                        eatoms_tensor[:nlocal].copy_(e_local.to(dtype=eatoms_tensor.dtype, device=eatoms_tensor.device))
            data.energy = float(e_local.sum().cpu())

        if hasattr(data, "update_pair_forces"):
            lammps_pair_forces = pair_forces if pair_forces.device.type == "cuda" else pair_forces.numpy()
            data.update_pair_forces(lammps_pair_forces)
            _log_cuda_memory("after force copy", self.device)
            _maybe_empty_cuda_cache(self.device, self._step)
            return

        f_view = data.f
        if f_view is None:
            raise RuntimeError("ML-IAP data.f is NULL: Cannot write forces.")
        pair_i = torch.as_tensor(data.pair_i, dtype=torch.long, device=pair_forces.device)[:npairs]
        pair_j = torch.as_tensor(data.pair_j, dtype=torch.long, device=pair_forces.device)[:npairs]
        forces = torch.zeros(ntotal, 3, dtype=pair_forces.dtype, device=pair_forces.device)
        forces.index_add_(0, pair_i, pair_forces)
        forces.index_add_(0, pair_j, -pair_forces)
        if torch.is_tensor(f_view):
            f_view[:ntotal].copy_(forces[:ntotal].to(f_view.dtype).to(f_view.device))
        else:
            np.asarray(f_view)[:ntotal, :] = forces[:ntotal].cpu().numpy()
        _log_cuda_memory("after force copy", self.device)
        _maybe_empty_cuda_cache(self.device, self._step)


def _has_mliap_attr(data, name: str) -> bool:
    """Check for an ML-IAP attribute without evaluating its descriptor.

    Args:
        data: Object whose class and instance dictionary are inspected.
        name (str): Attribute name.

    Returns:
        bool: Whether the class defines the attribute or the instance dictionary
            contains its name.
    """
    return hasattr(type(data), name) or name in getattr(data, "__dict__", {})


def build_lammps_unified(
    checkpoint: Union[str, Path],
    compile_mode: Optional[str] = None,
) -> FlashCartLAMMPSUnified:
    """Load an inference checkpoint on CPU and wrap it for LAMMPS.

    Setting ``compile_mode`` records how to evaluate the model in LAMMPS. It does
    not compile the model here.

    Args:
        checkpoint (Union[str, Path]): Inference checkpoint directory.
        compile_mode (str, optional): ``torch.compile`` mode, as described in
            :class:`FlashCartLAMMPSUnified`. Default: None.

    Returns:
        FlashCartLAMMPSUnified: Interface initialized with the saved model.
    """
    model = FlashCartPotential.from_checkpoint(checkpoint, device=torch.device("cpu"), map_location="cpu")
    return FlashCartLAMMPSUnified(model, compile_mode=compile_mode)


def save_lammps_unified(
    unified: FlashCartLAMMPSUnified,
    path: Union[str, Path],
) -> Path:
    """Serialize a FlashCart ML-IAP interface for loading by LAMMPS.

    The file contains the interface and its model, saved with ``torch.save``, for use
    with the LAMMPS ``mliap unified`` command. Parent directories are created if needed.
    Compilation settings are saved, but cached compiled functions are omitted. They
    are recreated on evaluation after loading when compilation is enabled.

    Args:
        unified (FlashCartLAMMPSUnified): Interface to save.
        path (Union[str, Path]): Output file path.

    Returns:
        Path: Path to the saved interface.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(unified, path)
    return path
