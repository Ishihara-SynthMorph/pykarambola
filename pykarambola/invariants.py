"""
SO(3) and O(3) invariant scalar construction from Minkowski tensors.

This module computes rotational invariants from arbitrary combinations of
rank 0, 1, and 2 tensors. The tensor set is not fixed — users may pass any
subset or custom tensors, and the invariant enumeration adapts dynamically.

The invariants are organized by polynomial degree:
- Degree 1: Scalars (rank-0 tensors and traces of rank-2 tensors)
- Degree 2: Dot products (vectors) and Frobenius inner products (matrices)
- Degree 3: Quadratic forms, triple traces, and pseudo-scalars

References
----------
Geiger & Smidt (2022) - e3nn: Euclidean Neural Networks
Schroeder-Turk et al. (2011) - Minkowski Tensors of Anisotropic Spatial Structure
"""

from __future__ import annotations

from itertools import combinations, combinations_with_replacement
from typing import Literal

import numpy as np


# Known linear dependencies: Tr(w102)/3 = w100, Tr(w202)/3 = w200
# Only applied when both tensors are present in the input.
_KNOWN_TRACE_DEPS = {
    'w102': 'w100',
    'w202': 'w200',
}


def _infer_rank(tensor) -> int:
    """Infer tensor rank from its shape.

    Parameters
    ----------
    tensor : scalar, array-like
        A rank-0 (scalar), rank-1 (3-vector), or rank-2 (3x3 matrix) tensor.

    Returns
    -------
    int
        The tensor rank: 0, 1, or 2.

    Raises
    ------
    ValueError
        If the tensor shape is not supported.
    """
    if isinstance(tensor, (int, float)):
        return 0
    arr = np.asarray(tensor)
    if arr.ndim == 0:
        return 0
    if arr.shape == (3,):
        return 1
    if arr.shape == (3, 3):
        return 2
    raise ValueError(f"Unsupported tensor shape: {arr.shape}. Expected scalar, (3,), or (3,3).")


def _trace_rank2(M: np.ndarray) -> float:
    """Compute the 0e (scalar) component of a rank-2 tensor: Tr(M) / 3."""
    return np.trace(M) / 3.0


def _traceless_rank2(M: np.ndarray) -> np.ndarray:
    """Compute the 2e (traceless) component of a rank-2 tensor: M - (Tr(M)/3) I."""
    trace_over_3 = np.trace(M) / 3.0
    return M - trace_over_3 * np.eye(3)


def decompose_all(tensors_dict: dict[str, np.ndarray | float]) -> dict[tuple[str, str], np.ndarray | float]:
    """Decompose tensors into irreducible representations (irreps).

    For each input tensor, decomposes into SO(3) irreps labeled by angular
    momentum quantum number and parity:
    - Rank 0 (scalar): '0e' (even parity scalar)
    - Rank 1 (vector): '1e' (odd parity vector, but even under proper rotations)
    - Rank 2 (matrix): '0e' (trace/3) + '2e' (traceless symmetric part)

    Parameters
    ----------
    tensors_dict : dict
        Mapping from tensor name to tensor value. Ranks are inferred from shape.

    Returns
    -------
    dict[tuple[str, str], np.ndarray | float]
        Mapping from (tensor_name, irrep_label) to the irrep component.
        For rank-2 tensors, both '0e' (scalar trace) and '2e' (traceless matrix)
        are returned.

    Examples
    --------
    >>> decomposed = decompose_all({'w000': 1.0, 'w010': np.array([1, 0, 0]), 'w020': np.eye(3)})
    >>> decomposed[('w000', '0e')]
    1.0
    >>> decomposed[('w010', '1e')]
    array([1., 0., 0.])
    >>> decomposed[('w020', '0e')]  # trace/3
    1.0
    >>> decomposed[('w020', '2e')]  # traceless part
    array([[0., 0., 0.],
           [0., 0., 0.],
           [0., 0., 0.]])
    """
    result = {}
    for name, tensor in tensors_dict.items():
        rank = _infer_rank(tensor)
        if rank == 0:
            val = float(tensor) if isinstance(tensor, (int, float)) else float(np.asarray(tensor))
            result[(name, '0e')] = val
        elif rank == 1:
            result[(name, '1e')] = np.asarray(tensor, dtype=float)
        elif rank == 2:
            M = np.asarray(tensor, dtype=float)
            result[(name, '0e')] = _trace_rank2(M)
            result[(name, '2e')] = _traceless_rank2(M)
    return result


def _collect_by_irrep(decomposed: dict[tuple[str, str], np.ndarray | float], irrep: str) -> list[tuple[str, np.ndarray | float]]:
    """Collect all components matching a given irrep label, sorted by tensor name."""
    items = [(name, val) for (name, irrep_label), val in decomposed.items() if irrep_label == irrep]
    return sorted(items, key=lambda x: x[0])


def _degree1_scalars(
    decomposed: dict[tuple[str, str], np.ndarray | float],
    deduplicate: bool = True,
) -> dict[str, float]:
    """Collect degree-1 invariants (all 0e scalars).

    Parameters
    ----------
    decomposed : dict
        Output from decompose_all().
    deduplicate : bool
        If True, remove scalars that are linearly dependent on others
        (e.g., Tr(w102)/3 when w100 is also present).

    Returns
    -------
    dict[str, float]
        Mapping from invariant label to scalar value.
    """
    scalars = _collect_by_irrep(decomposed, '0e')
    result = {}

    # Track which tensor names are present (for dependency checking)
    tensor_names = {name for name, _ in scalars}

    for name, val in scalars:
        # Check if this is a trace that duplicates another scalar
        if deduplicate and name in _KNOWN_TRACE_DEPS:
            base_scalar = _KNOWN_TRACE_DEPS[name]
            if base_scalar in tensor_names:
                # Skip this trace — it's linearly dependent on the base scalar
                continue
        result[name] = float(val)

    return result


def _vector_dot_products(decomposed: dict[tuple[str, str], np.ndarray | float]) -> dict[str, float]:
    """Compute degree-2 invariants: dot products of all 1e vectors.

    Returns C(n,2) + n = n(n+1)/2 invariants for n vectors.
    """
    vectors = _collect_by_irrep(decomposed, '1e')
    result = {}
    for i, (name_i, vi) in enumerate(vectors):
        for name_j, vj in vectors[i:]:
            label = f"dot_{name_i}_{name_j}"
            result[label] = float(np.dot(vi, vj))
    return result


def _frobenius_inner_products(decomposed: dict[tuple[str, str], np.ndarray | float]) -> dict[str, float]:
    """Compute degree-2 invariants: Frobenius inner products of all 2e matrices.

    Returns C(m,2) + m = m(m+1)/2 invariants for m traceless matrices.
    """
    matrices = _collect_by_irrep(decomposed, '2e')
    result = {}
    for i, (name_i, Ti) in enumerate(matrices):
        for name_j, Tj in matrices[i:]:
            label = f"frob_{name_i}_{name_j}"
            result[label] = float(np.einsum('ij,ij->', Ti, Tj))
    return result


def _quadratic_forms(decomposed: dict[tuple[str, str], np.ndarray | float]) -> dict[str, float]:
    """Compute degree-3 O(3) invariants: v_i^T T_k v_j for all matrices and vector pairs.

    For symmetric matrices, v_i^T T v_j = v_j^T T v_i, so we only compute i <= j.
    """
    vectors = _collect_by_irrep(decomposed, '1e')
    matrices = _collect_by_irrep(decomposed, '2e')
    result = {}
    for name_T, T in matrices:
        for i, (name_i, vi) in enumerate(vectors):
            for name_j, vj in vectors[i:]:
                label = f"qf_{name_i}_{name_T}_{name_j}"
                result[label] = float(np.einsum('i,ij,j->', vi, T, vj))
    return result


def _triple_traces(decomposed: dict[tuple[str, str], np.ndarray | float]) -> dict[str, float]:
    """Compute degree-3 O(3) invariants: Tr(T_i T_j T_k) for all matrix multisets.

    Uses combinations_with_replacement to enumerate multisets {i, j, k}.
    """
    matrices = _collect_by_irrep(decomposed, '2e')
    if len(matrices) == 0:
        return {}

    result = {}
    names = [name for name, _ in matrices]
    mat_dict = {name: M for name, M in matrices}

    for combo in combinations_with_replacement(range(len(matrices)), 3):
        name_i, name_j, name_k = names[combo[0]], names[combo[1]], names[combo[2]]
        Ti, Tj, Tk = mat_dict[name_i], mat_dict[name_j], mat_dict[name_k]
        label = f"ttr_{name_i}_{name_j}_{name_k}"
        result[label] = float(np.einsum('ij,jk,ki->', Ti, Tj, Tk))
    return result


def _triple_vector_dets(decomposed: dict[tuple[str, str], np.ndarray | float]) -> dict[str, float]:
    """Compute degree-3 SO(3)-only pseudo-scalars: det([v_i, v_j, v_k]) for i < j < k.

    These flip sign under reflections (improper rotations).
    """
    vectors = _collect_by_irrep(decomposed, '1e')
    if len(vectors) < 3:
        return {}

    result = {}
    for (name_i, vi), (name_j, vj), (name_k, vk) in combinations(vectors, 3):
        label = f"det_{name_i}_{name_j}_{name_k}"
        mat = np.column_stack([vi, vj, vk])
        result[label] = float(np.linalg.det(mat))
    return result


def _commutator_pseudoscalars(decomposed: dict[tuple[str, str], np.ndarray | float]) -> dict[str, float]:
    """Compute degree-3 SO(3)-only pseudo-scalars from matrix commutators.

    For matrices T_a, T_b (a < b) and vector v_k:
    Extracts the axial vector from [T_a, T_b] and dots with v_k.

    The axial vector of an antisymmetric matrix A is:
    a = (A_{32}, A_{13}, A_{21}) such that A_{ij} = epsilon_{ijk} a_k
    """
    vectors = _collect_by_irrep(decomposed, '1e')
    matrices = _collect_by_irrep(decomposed, '2e')

    if len(matrices) < 2 or len(vectors) == 0:
        return {}

    result = {}
    for (name_a, Ta), (name_b, Tb) in combinations(matrices, 2):
        # Commutator [Ta, Tb] = Ta @ Tb - Tb @ Ta (antisymmetric)
        comm = Ta @ Tb - Tb @ Ta
        # Extract axial vector: (C_{32}, C_{13}, C_{21}) = (C[2,1]-C[1,2], C[0,2]-C[2,0], C[1,0]-C[0,1]) / 2
        # For antisymmetric matrix: C[i,j] = -C[j,i], so axial = (C[2,1], C[0,2], C[1,0])
        axial = np.array([comm[2, 1], comm[0, 2], comm[1, 0]])

        for name_k, vk in vectors:
            label = f"comm_{name_a}_{name_b}_{name_k}"
            result[label] = float(np.dot(axial, vk))

    return result


def compute_invariants(
    tensors_dict: dict[str, np.ndarray | float],
    max_degree: int = 3,
    symmetry: Literal['O3', 'SO3'] = 'SO3',
    deduplicate_scalars: bool = True,
) -> dict[str, float]:
    """Compute SO(3) or O(3) invariants from arbitrary Minkowski tensors.

    This function computes a complete basis of polynomial invariants up to
    the specified degree. The tensor set is flexible — any combination of
    rank 0, 1, and 2 tensors can be provided.

    Parameters
    ----------
    tensors_dict : dict[str, np.ndarray | float]
        Mapping from tensor name to tensor value. Tensor ranks are inferred
        from their shapes:
        - scalar (int, float, 0-d array) -> rank 0
        - shape (3,) -> rank 1 (vector)
        - shape (3, 3) -> rank 2 (matrix)

    max_degree : int, default=3
        Maximum polynomial degree of invariants to compute:
        - 1: Only degree-1 (scalars and traces)
        - 2: Add degree-2 (dot products, Frobenius inner products)
        - 3: Add degree-3 (quadratic forms, triple traces, pseudo-scalars)

    symmetry : {'O3', 'SO3'}, default='SO3'
        Symmetry group for invariants:
        - 'O3': Only true scalars (invariant under rotations AND reflections)
        - 'SO3': Include pseudo-scalars (flip sign under reflections)

    deduplicate_scalars : bool, default=True
        If True, remove degree-1 scalars that are linearly dependent on others.
        Specifically, Tr(w102)/3 is removed when w100 is present, and
        Tr(w202)/3 is removed when w200 is present.

    Returns
    -------
    dict[str, float]
        Mapping from invariant label to computed value. Labels are deterministic
        and sorted lexicographically by tensor name.

    Notes
    -----
    Translation invariance: Many Minkowski tensors depend on the choice of
    origin/centroid. The invariants computed here will change if the mesh
    is translated, unless the tensors themselves are translation-invariant.

    The invariant labels follow these patterns:
    - Degree 1: tensor name (e.g., 'w000', 'w020')
    - Degree 2: 'dot_{v1}_{v2}', 'frob_{T1}_{T2}'
    - Degree 3: 'qf_{v1}_{T}_{v2}', 'ttr_{T1}_{T2}_{T3}',
                'det_{v1}_{v2}_{v3}', 'comm_{T1}_{T2}_{v}'

    Examples
    --------
    >>> import numpy as np
    >>> tensors = {
    ...     'w000': 1.0,
    ...     'w010': np.array([1, 0, 0]),
    ...     'w020': np.eye(3) * 0.5,
    ... }
    >>> inv = compute_invariants(tensors, max_degree=2)
    >>> 'w000' in inv
    True
    >>> 'dot_w010_w010' in inv
    True
    """
    if not tensors_dict:
        return {}

    # Decompose all tensors into irreps
    decomposed = decompose_all(tensors_dict)

    result = {}

    # Degree 1: scalars
    result.update(_degree1_scalars(decomposed, deduplicate=deduplicate_scalars))

    if max_degree >= 2:
        # Degree 2: bilinear invariants
        result.update(_vector_dot_products(decomposed))
        result.update(_frobenius_inner_products(decomposed))

    if max_degree >= 3:
        # Degree 3 O(3) invariants (true scalars)
        result.update(_quadratic_forms(decomposed))
        result.update(_triple_traces(decomposed))

        # Degree 3 SO(3)-only pseudo-scalars
        if symmetry == 'SO3':
            result.update(_triple_vector_dets(decomposed))
            result.update(_commutator_pseudoscalars(decomposed))

    return result
