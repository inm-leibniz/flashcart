"""Describe tensor layouts, convert Cartesian components, and select kernel precision."""

import itertools
import functools

from typing import List, Literal, Tuple

import torch

TritonDotInputPrecision = Literal["ieee", "tf32", "tf32x3"]


@torch._dynamo.assume_constant_result
def get_triton_dot_input_precision() -> TritonDotInputPrecision:
    """Select Triton dot precision from the PyTorch matrix-product settings.

    The result is treated as constant during ``torch.compile``.

    Returns:
        str: ``"tf32"`` when CUDA is available, TF32 matrix products are enabled, and
            the float32 matrix-product precision is ``"high"``. Otherwise, return
            ``"ieee"``.
    """
    if not torch.cuda.is_available():
        return "ieee"
    if not torch.backends.cuda.matmul.allow_tf32:
        return "ieee"
    mode = torch.get_float32_matmul_precision()
    if mode == "highest":
        return "ieee"
    if mode == "high":
        return "tf32"
    return "ieee"


def get_linear_triton_kernel_options(
    dtype: torch.dtype,
) -> tuple[bool, TritonDotInputPrecision]:
    """Select the accumulator mode and dot precision for the linear kernels.

    Args:
        dtype (torch.dtype): Input dtype.

    Returns:
        tuple[bool, str]: Whether to use float64 accumulation and the Triton dot-input
            precision. Float64 inputs return ``(True, "ieee")``.
    """
    if dtype == torch.float64:
        return True, "ieee"
    return False, get_triton_dot_input_precision()


def _full_to_stored_matrix(l: int) -> torch.Tensor:
    """Build the matrix that reconstructs full Cartesian components.

    Symmetry and tracelessness express each full component as a linear
    combination of the stored independent components. Despite the function
    name, the matrix maps stored components to full Cartesian components.

    Args:
        l (int): Nonnegative tensor rank.

    Returns:
        torch.Tensor: Reconstruction matrix of shape ``(3**l, 2*l + 1)``, using
            the default floating-point dtype and device. Rows follow flattened
            Cartesian index order, with the last index varying fastest.
    """
    from flashcart.o3._codegen_common import stored_basis

    if l == 0:
        return torch.tensor([[1.0]])
    basis = stored_basis(l)
    basis_index = {tup: i for i, tup in enumerate(basis)}
    dim_in = 2 * l + 1
    cache: dict = {}

    def get_sorted(a: int, b: int, c: int) -> torch.Tensor:
        if (a, b, c) in cache:
            return cache[(a, b, c)]
        if (a, b, c) in basis_index:
            v = torch.zeros(dim_in)
            v[basis_index[(a, b, c)]] = 1.0
        elif c >= 2:
            v = -get_sorted(a + 2, b, c - 2) - get_sorted(a, b + 2, c - 2)
        else:
            raise ValueError(f"Unexpected multi-index ({a}, {b}, {c}) for l={l}.")
        cache[(a, b, c)] = v
        return v

    M = torch.empty(3**l, dim_in)
    for flat_idx, multi_idx in enumerate(itertools.product(range(3), repeat=l)):
        a = multi_idx.count(0)
        b = multi_idx.count(1)
        c = multi_idx.count(2)
        M[flat_idx] = get_sorted(a, b, c)
    return M


@functools.lru_cache(maxsize=None)
def _stored_full_indices(l: int) -> Tuple[int, ...]:
    """Locate the independent components in a full Cartesian tensor.

    The result is cached by tensor rank.

    Args:
        l (int): Nonnegative tensor rank.

    Returns:
        tuple[int, ...]: Indices into the flattened Cartesian component axis,
            in the order defined by ``stored_basis``. The last Cartesian
            index varies fastest in the full representation.
    """
    from flashcart.o3._codegen_common import stored_basis

    if l == 0:
        return (0,)
    out: List[int] = []
    for a, b, c in stored_basis(l):
        multi = (0,) * a + (1,) * b + (2,) * c
        flat = 0
        for d in multi:
            flat = flat * 3 + d
        out.append(flat)
    return tuple(out)


def apply_path_weight(
    x: torch.Tensor,
    weights: torch.Tensor,
    path_idx: int,
    out_features: int,
    shared: bool,
) -> torch.Tensor:
    """Scale one path's output block by its per-feature weights.

    Args:
        x (torch.Tensor): Path output of shape ``(batch, ..., out_features)``.
        weights (torch.Tensor): Flat weights over (path, feature). One-dimensional when
            shared, and of shape ``(batch, n)`` otherwise.
        path_idx (int): Which path's weight block to apply.
        out_features (int): Features per path.
        shared (bool): Weights are batch-shared (1-D).

    Returns:
        torch.Tensor: Weighted path features with the same shape as ``x``.
    """
    start = path_idx * out_features
    end = start + out_features
    if shared:
        w = weights[start:end]
    else:
        w = weights[:, start:end]
        w = w.reshape(w.shape[0], *([1] * (x.dim() - 2)), w.shape[1])
    return x * w


def count_output_paths(
    in1_l_max: int, in2_l_max: int, out_l_max: int, symmetric_product: bool = False
) -> Tuple[List[int], int]:
    """Count the allowed coupling paths at each output tensor rank.

    A path satisfies ``abs(l1 - l2) <= l_out <= l1 + l2`` and an even value of
    ``l1 + l2 + l_out``. At each output rank, paths with ``l1 >= l2`` are ordered by
    increasing ``l1`` and then ``l2``. For nonsymmetric products, exchanged unequal-rank
    paths follow in the same order. Tensor-product weights use this ordering.

    Args:
        in1_l_max (int): Maximum tensor rank of the first input.
        in2_l_max (int): Maximum tensor rank of the second input.
        out_l_max (int): Maximum tensor rank of the output.
        symmetric_product (bool, optional): Count only unordered pairs (identical
            inputs). Default: False.

    Returns:
        tuple[list[int], int]: Path counts for output ranks zero through ``out_l_max``,
            followed by their sum.
    """
    from flashcart.o3._codegen_tensor_product import even_tp_path_allowed

    out_paths = [0 for _ in range(out_l_max + 1)]
    l_pair_max = max(in1_l_max, in2_l_max)

    for l_out in range(out_l_max + 1):
        for l1 in range(l_pair_max + 1):
            for l2 in range(l1 + 1):
                if not even_tp_path_allowed(l1, l2, l_out):
                    continue
                if l1 > in1_l_max or l2 > in2_l_max:
                    continue
                out_paths[l_out] += 1

        if not symmetric_product:
            for l1 in range(l_pair_max + 1):
                for l2 in range(l1 + 1):
                    if l1 == l2:
                        continue
                    if not even_tp_path_allowed(l1, l2, l_out):
                        continue
                    if l1 > in2_l_max or l2 > in1_l_max:
                        continue
                    out_paths[l_out] += 1

    return out_paths, sum(out_paths)


def get_irreps_slices(l_max: int, n_features: int, n_paths: List[int] = None) -> list:
    """Locate tensor-rank blocks in the flattened stored-component layout.

    Args:
        l_max (int): Maximum tensor rank.
        n_features (int): Feature channels per tensor rank.
        n_paths (list[int], optional): Paths per tensor rank. Default: one each.

    Returns:
        list[tuple[int, int]]: Start and exclusive stop offsets for each tensor rank, in
            increasing order.
    """
    if n_paths is None:
        n_paths = [1 for _ in range(l_max + 1)]

    slices = []
    start = 0
    for l in range(l_max + 1):
        size = (2 * l + 1) * n_features * n_paths[l]
        slices.append((start, start + size))
        start += size
    return slices


ShapeMode = Literal["flattened", "factorized"]


def get_irreps_shapes(
    l_max: int,
    n_features: int,
    n_paths: List[int] = None,
    mode: ShapeMode = "flattened",
) -> list:
    """Return shape tuples for blocks at each tensor rank.

    With ``mode="factorized"``, the axes are component, feature, and path.
    Tensor-product outputs instead store component, path, and feature. The
    factorized shape is therefore not a direct reshape specification for a
    tensor-product output when both path and feature counts exceed one.

    Args:
        l_max (int): Maximum tensor rank.
        n_features (int): Feature channels per tensor rank.
        n_paths (list[int], optional): Paths per tensor rank. Default: one each.
        mode (str, optional): ``"flattened"`` gives
            ``(2*l + 1, n_features * n_paths[l])``. ``"factorized"`` gives
            ``(2*l + 1, n_features, n_paths[l])``. Default: ``"flattened"``.

    Returns:
        list[tuple[int, ...]]: Block shapes in increasing tensor rank, with axes
            determined by ``mode`` and without a batch dimension.
    """
    if n_paths is None:
        n_paths = [1] * (l_max + 1)
    if mode == "factorized":
        return [(2 * l + 1, n_features, n_paths[l]) for l in range(l_max + 1)]
    return [(2 * l + 1, n_features * n_paths[l]) for l in range(l_max + 1)]


def get_cartesian_slices(l_max: int, n_features: int, n_paths: List[int] = None) -> list:
    """Locate tensor-rank blocks in the flattened full Cartesian layout.

    Args:
        l_max (int): Maximum tensor rank.
        n_features (int): Feature channels per tensor rank.
        n_paths (list[int], optional): Paths per tensor rank. Default: one each.

    Returns:
        list[tuple[int, int]]: Start and exclusive stop offsets for each tensor rank, in
            increasing order.
    """
    if n_paths is None:
        n_paths = [1 for _ in range(l_max + 1)]

    slices = []
    start = 0
    for l in range(l_max + 1):
        size = (3**l) * n_features * n_paths[l]
        slices.append((start, start + size))
        start += size
    return slices


def irreps_to_cartesian(irreps: torch.Tensor, l_max: int, n_paths: List[int] = None):
    """Reconstruct full Cartesian tensors from their independent components.

    Symmetry and tracelessness determine the remaining components at each tensor rank.
    The stored basis is not orthonormal for ranks two and above, so the ordinary
    Euclidean norm of the stored components does not generally equal the full
    Cartesian tensor norm. Paths and features are treated as a combined trailing
    axis whose ordering is preserved.

    Args:
        irreps (torch.Tensor): Features of shape ``(batch, dim)`` in the stored layout.
        l_max (int): Maximum tensor rank.
        n_paths (list[int], optional): Paths per tensor rank. Default: one each.

    Returns:
        torch.Tensor: Full Cartesian components of shape
            ``(batch, sum(3**l * n_features * n_paths[l]))``, summed over ranks zero
            through ``l_max``. The feature count is inferred from the input width.

    Raises:
        ValueError: The input width is incompatible with the requested tensor
            ranks and path counts.
    """
    batch_size = irreps.shape[0]
    total_size = irreps.shape[1]

    if n_paths is None:
        n_paths = [1 for _ in range(l_max + 1)]
        irreps_dim = (l_max + 1) ** 2
    else:
        irreps_dim = sum((2 * l + 1) * n_paths[l] for l in range(l_max + 1))

    if total_size % irreps_dim != 0:
        raise ValueError(f"Tensor size {total_size} incompatible with l_max={l_max}.")
    n_features = total_size // irreps_dim

    parts: List[torch.Tensor] = []
    comp_idx = 0
    for l in range(l_max + 1):
        dim_l = 2 * l + 1
        block_size = dim_l * n_features * n_paths[l]
        block = irreps[:, comp_idx : comp_idx + block_size]
        comp_idx += block_size

        if l == 0:
            parts.append(block)
            continue

        stored = block.reshape(batch_size, dim_l, n_features * n_paths[l])
        M = _full_to_stored_matrix(l).to(device=block.device, dtype=block.dtype)
        full = torch.einsum("fk,bku->bfu", M, stored)
        parts.append(full.reshape(batch_size, -1))

    return torch.cat(parts, dim=-1)


def cartesian_to_irreps(tensors: torch.Tensor, l_max: int, n_paths: List[int] = None):
    """Select the independent components of full Cartesian tensors.

    This operation inverts ``irreps_to_cartesian`` for symmetric traceless inputs. It
    does not project arbitrary inputs onto the symmetric traceless subspace.
    Paths and features are treated as a combined trailing axis whose ordering
    is preserved.

    Args:
        tensors (torch.Tensor): Symmetric traceless tensor features of shape
            ``(batch, dim)`` in the full Cartesian layout.
        l_max (int): Maximum tensor rank.
        n_paths (list[int], optional): Paths per tensor rank. Default: one each.

    Returns:
        torch.Tensor: Stored components of shape
            ``(batch, sum((2*l + 1) * n_features * n_paths[l]))``, summed over ranks
            zero through ``l_max``. The feature count is inferred from the input width.

    Raises:
        ValueError: The input width is incompatible with the requested tensor
            ranks and path counts.
    """
    batch_size = tensors.shape[0]
    total_size = tensors.shape[1]

    if n_paths is None:
        n_paths = [1 for _ in range(l_max + 1)]
        cartesian_dim = sum(3**l for l in range(l_max + 1))
    else:
        cartesian_dim = sum((3**l) * n_paths[l] for l in range(l_max + 1))

    if total_size % cartesian_dim != 0:
        raise ValueError(f"Tensor size {total_size} incompatible with l_max={l_max}.")
    n_features = total_size // cartesian_dim

    parts: List[torch.Tensor] = []
    comp_idx = 0
    for l in range(l_max + 1):
        block_size = (3**l) * n_features * n_paths[l]
        block = tensors[:, comp_idx : comp_idx + block_size]
        comp_idx += block_size

        if l == 0:
            parts.append(block)
            continue

        full = block.reshape(batch_size, 3**l, n_features * n_paths[l])
        idx = torch.as_tensor(_stored_full_indices(l), device=full.device, dtype=torch.long)
        stored = full.index_select(1, idx)
        parts.append(stored.reshape(batch_size, -1))

    return torch.cat(parts, dim=1)


def rotate_irreps(
    irreps: torch.Tensor,
    rot_matrix: torch.Tensor,
    l_max: int,
    n_paths: List[int] = None,
) -> torch.Tensor:
    """Apply an orthogonal transformation to stored Cartesian tensor components.

    The tensors are reconstructed in full Cartesian form, each tensor index is
    contracted with the rotation matrix, and the independent components are selected
    again. For rank-one features, the convention is ``v_rotated = rot_matrix @ v``.

    Args:
        irreps (torch.Tensor): Features of shape ``(batch, dim)`` in the stored layout.
        rot_matrix (torch.Tensor): Orthogonal transformation of shape ``(3, 3)``,
            shared by all input rows and using the input dtype and device.
            Rotations and reflections are supported. Orthogonality is not checked.
        l_max (int): Maximum tensor rank. Ranks above 13 are not supported.
        n_paths (list[int], optional): Paths per tensor rank. Default: one each.

    Returns:
        torch.Tensor: Rotated features with the same shape and stored-component layout
            as ``irreps``.

    Raises:
        ValueError: The input width is incompatible with the requested layout,
            or ``rot_matrix`` does not have shape ``(3, 3)``.
        NotImplementedError: ``l_max`` exceeds 13.
    """
    batch_size = irreps.shape[0]
    total_size = irreps.shape[1]

    if n_paths is None:
        n_paths = [1 for _ in range(l_max + 1)]
        irreps_dim = (l_max + 1) ** 2
    else:
        irreps_dim = sum([(2 * l + 1) * n_paths[l] for l in range(l_max + 1)])

    if total_size % irreps_dim != 0:
        raise ValueError(f"Tensor size {total_size} incompatible with l_max={l_max}.")
    n_features = total_size // irreps_dim

    if rot_matrix.dim() != 2 or rot_matrix.shape[0] != 3 or rot_matrix.shape[1] != 3:
        raise ValueError(f"Rotation matrix must be of shape [3, 3]. Provided shape: {rot_matrix.shape}.")

    cartesian = irreps_to_cartesian(irreps, l_max, n_paths)

    rotated_parts: List[torch.Tensor] = []
    comp_idx = 0

    for l in range(l_max + 1):
        n_paths_l = n_paths[l]
        cart_size = (3**l) * n_features * n_paths_l

        cart_l = cartesian[:, comp_idx : comp_idx + cart_size]
        comp_idx += cart_size

        if l == 0:
            rotated_parts.append(cart_l)
            continue

        T = cart_l.reshape(batch_size, *([3] * l), n_features * n_paths_l)
        T_rot = _rotate_full_cartesian(T, rot_matrix, l)
        rotated_parts.append(T_rot.reshape(batch_size, -1))

    rotated_cartesian = torch.cat(rotated_parts, dim=1)
    rotated_irreps = cartesian_to_irreps(rotated_cartesian, l_max, n_paths)

    return rotated_irreps


def _rotate_full_cartesian(T: torch.Tensor, R: torch.Tensor, l: int) -> torch.Tensor:
    """Transform every Cartesian index with the same orthogonal matrix.

    Args:
        T (torch.Tensor): Full tensor of shape ``(batch, 3, ..., 3, features)``,
            with ``l`` Cartesian axes.
        R (torch.Tensor): Matrix of shape ``(3, 3)``, using the input dtype and device.
        l (int): Tensor rank, at most 13.

    Returns:
        torch.Tensor: Transformed tensor with the same shape as ``T``. Rank-zero
            inputs are returned unchanged.
    """
    if l == 0:
        return T
    if l > 13:
        raise NotImplementedError(
            f"_rotate_full_cartesian uses dynamic einsum subscripts and only " f"supports l <= 13, got l={l}."
        )
    out_letters = "abcdefghijklm"[:l]
    in_letters = "nopqrstuvwxyz"[:l]
    R_terms = [f"{o}{i}" for o, i in zip(out_letters, in_letters)]
    T_term = "B" + in_letters + "F"
    out_term = "B" + out_letters + "F"
    spec = ",".join(R_terms) + "," + T_term + "->" + out_term
    return torch.einsum(spec, *([R] * l), T)
