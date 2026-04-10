"""
SO(3), O(3), and SO(2) invariant scalar construction from Minkowski tensors.

This module computes rotational invariants from arbitrary combinations of
rank 0, 1, and 2 tensors. The tensor set is not fixed — users may pass any
subset or custom tensors, and the invariant enumeration adapts dynamically.

The invariants are organized by polynomial degree:
- Degree 1: Scalars (rank-0 tensors and traces of rank-2 tensors)
- Degree 2: Dot products (vectors) and Frobenius inner products (matrices)
- Degree 3: Quadratic forms, triple traces, and pseudo-scalars

Note: This module labels all l=1 irreps as '1e'. Unlike e3nn which distinguishes
'1e' (axial) from '1o' (polar), here the label indicates only the angular
momentum quantum number. Parity is handled via the symmetry='O3'/'SO3' parameter.

For symmetry='SO2', components are decomposed by their charge m under rotation
about the z-axis: m=0 scalars, |m|=1 doublets, and |m|=2 doublets.

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
    """Compute the 2e (traceless symmetric) component of a rank-2 tensor.

    The input is first symmetrized, then the traceless part is extracted.
    """
    M_sym = (M + M.T) / 2.0
    trace_over_3 = np.trace(M_sym) / 3.0
    return M_sym - trace_over_3 * np.eye(3)


def decompose_all(tensors_dict: dict[str, np.ndarray | float]) -> dict[tuple[str, str], np.ndarray | float]:
    """Decompose tensors into irreducible representations (irreps).

    For each input tensor, decomposes into SO(3) irreps labeled by angular
    momentum quantum number and parity:
    - Rank 0 (scalar): '0e' (even parity scalar)
    - Rank 1 (vector): '1e' (polar vector, l=1 irrep)
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
    a = (A_{12}, A_{20}, A_{01}) such that A_{ij} = epsilon_{ijk} a_k
    """
    vectors = _collect_by_irrep(decomposed, '1e')
    matrices = _collect_by_irrep(decomposed, '2e')

    if len(matrices) < 2 or len(vectors) == 0:
        return {}

    result = {}
    for (name_a, Ta), (name_b, Tb) in combinations(matrices, 2):
        # Commutator [Ta, Tb] = Ta @ Tb - Tb @ Ta (antisymmetric)
        comm = Ta @ Tb - Tb @ Ta
        # Extract axial vector: (C_{12}, C_{20}, C_{01})
        # For antisymmetric matrix: C[i,j] = -C[j,i], so axial = (C[1,2], C[2,0], C[0,1])
        axial = np.array([comm[1, 2], comm[2, 0], comm[0, 1]])

        for name_k, vk in vectors:
            label = f"comm_{name_a}_{name_b}_{name_k}"
            result[label] = float(np.dot(axial, vk))

    return result


def _decompose_so2(tensors_dict: dict[str, np.ndarray | float]) -> dict[tuple[str, str], object]:
    """Decompose tensors into SO(2) components (charges under z-rotation).

    Returns entries keyed by (name, label) where label is one of:
    - 'sc'  : rank-0 scalar value (float)
    - 'z'   : v_z component of rank-1 vector (float)
    - 'xy'  : [v_x, v_y] doublet from rank-1 vector, |m|=1 (ndarray shape (2,))
    - 'tr'  : Tr(M)/3 isotropic scalar from rank-2 matrix (float)
    - 'tzz' : T_zz = M_zz - Tr(M)/3 traceless m=0 component (float)
    - 'xz'  : [M_xz, M_yz] doublet, |m|=1 (ndarray shape (2,))
    - 'm2'  : [M_xx - M_yy, 2*M_xy] doublet, |m|=2 (ndarray shape (2,))
    """
    result = {}
    for name, tensor in tensors_dict.items():
        rank = _infer_rank(tensor)
        if rank == 0:
            val = float(tensor) if isinstance(tensor, (int, float)) else float(np.asarray(tensor))
            result[(name, 'sc')] = val
        elif rank == 1:
            v = np.asarray(tensor, dtype=float)
            result[(name, 'z')] = float(v[2])
            result[(name, 'xy')] = np.array([v[0], v[1]])
        elif rank == 2:
            M = np.asarray(tensor, dtype=float)
            M_sym = (M + M.T) / 2.0
            tr_over3 = np.trace(M_sym) / 3.0
            result[(name, 'tr')] = float(tr_over3)
            result[(name, 'tzz')] = float(M_sym[2, 2] - tr_over3)
            result[(name, 'xz')] = np.array([M_sym[0, 2], M_sym[1, 2]])
            result[(name, 'm2')] = np.array([M_sym[0, 0] - M_sym[1, 1], 2.0 * M_sym[0, 1]])
    return result


def _so2_degree1_scalars(
    so2_dec: dict[tuple[str, str], object],
    deduplicate: bool = True,
) -> dict[str, float]:
    """Collect degree-1 SO(2) invariants: m=0 scalars.

    For rank-0: output key = name, value = S.
    For rank-1: output key = {name}_z, value = v_z.
    For rank-2: output key = name (dedup-able), value = Tr(M)/3;
                output key = {name}_zz, value = M_zz (always kept).
    """
    result = {}

    sc_names = sorted({name for (name, lbl) in so2_dec if lbl == 'sc'})
    z_names = sorted({name for (name, lbl) in so2_dec if lbl == 'z'})
    tr_names = sorted({name for (name, lbl) in so2_dec if lbl == 'tr'})

    def _set(key, val):
        if key in result:
            raise ValueError(
                f"SO(2) degree-1 key collision: '{key}' would be written twice. "
                "Rename tensors so that no rank-0 name ends with '_z' or '_zz' "
                "when a rank-1 or rank-2 tensor shares the prefix."
            )
        result[key] = val

    # rank-0 scalars
    for name in sc_names:
        _set(name, float(so2_dec[(name, 'sc')]))

    # rank-1 v_z
    for name in z_names:
        _set(f"{name}_z", float(so2_dec[(name, 'z')]))

    # rank-2: trace (dedup-able) + M_zz (always kept)
    for name in tr_names:
        tr_val = float(so2_dec[(name, 'tr')])
        tzz_val = float(so2_dec[(name, 'tzz')])

        # Skip trace key if linearly dependent on a present rank-0 scalar
        skip_trace = (
            deduplicate
            and name in _KNOWN_TRACE_DEPS
            and _KNOWN_TRACE_DEPS[name] in sc_names
        )
        if not skip_trace:
            _set(name, tr_val)

        # M_zz = tr + tzz; always included regardless of dedup
        _set(f"{name}_zz", tr_val + tzz_val)

    return result


def _so2_collect_doublets(
    so2_dec: dict[tuple[str, str], object],
    label: str,
) -> list[tuple[str, np.ndarray]]:
    """Collect all doublets of a given SO(2) label, sorted by compound label.

    Returns list of (compound_label, array) where compound_label = f"{name}_{label}".
    """
    items = [
        (f"{name}_{label}", val)
        for (name, lbl), val in so2_dec.items()
        if lbl == label
    ]
    return sorted(items, key=lambda x: x[0])


def _so2_doublet_inner_products(so2_dec: dict[tuple[str, str], object]) -> dict[str, float]:
    """Compute degree-2 SO(2) invariants: inner products of same-charge doublets.

    d1_{ci}_{cj}: dot product of two |m|=1 doublets (from rank-1 _xy and rank-2 _xz)
    d2_{ci}_{cj}: dot product of two |m|=2 doublets (from rank-2 _m2)
    """
    result = {}

    # |m|=1 doublets: both 'xy' (rank-1) and 'xz' (rank-2)
    m1_doublets = sorted(
        _so2_collect_doublets(so2_dec, 'xy') + _so2_collect_doublets(so2_dec, 'xz'),
        key=lambda x: x[0],
    )
    for i, (ci, di) in enumerate(m1_doublets):
        for cj, dj in m1_doublets[i:]:
            result[f"d1_{ci}_{cj}"] = float(np.dot(di, dj))

    # |m|=2 doublets: 'm2' from rank-2
    m2_doublets = _so2_collect_doublets(so2_dec, 'm2')
    for i, (ci, di) in enumerate(m2_doublets):
        for cj, dj in m2_doublets[i:]:
            result[f"d2_{ci}_{cj}"] = float(np.dot(di, dj))

    return result


def _so2_triple_products(so2_dec: dict[tuple[str, str], object]) -> dict[str, float]:
    """Compute degree-3 SO(2) invariants: triple couplings of |m|=1, |m|=1, |m|=2.

    For doublets a (|m|=1), b (|m|=1) with ca <= cb, and c (|m|=2):
        Re = (ax*bx - ay*by)*cx + (ax*by + ay*bx)*cy  = Re[conj(c) * (a*b)]
        Im = (ax*by + ay*bx)*cx - (ax*bx - ay*by)*cy  = Im[conj(c) * (a*b)]

    where a*b denotes complex multiplication: a = ax+i*ay, b = bx+i*by, c = cx+i*cy.

    These are genuinely new invariants not expressible as products of lower-degree terms.
    """
    m1_doublets = sorted(
        _so2_collect_doublets(so2_dec, 'xy') + _so2_collect_doublets(so2_dec, 'xz'),
        key=lambda x: x[0],
    )
    m2_doublets = _so2_collect_doublets(so2_dec, 'm2')

    if not m1_doublets or not m2_doublets:
        return {}

    result = {}
    for i, (ca, a) in enumerate(m1_doublets):
        for cb, b in m1_doublets[i:]:
            # Complex product a*b: (ax+i*ay)*(bx+i*by)
            re_ab = a[0] * b[0] - a[1] * b[1]  # Re(a*b)
            im_ab = a[0] * b[1] + a[1] * b[0]  # Im(a*b)
            for cc, c in m2_doublets:
                re_val = re_ab * c[0] + im_ab * c[1]   # Re[conj(c) * (a*b)]
                im_val = im_ab * c[0] - re_ab * c[1]   # Im[conj(c) * (a*b)]
                result[f"tp_re_{ca}_{cb}_{cc}"] = float(re_val)
                result[f"tp_im_{ca}_{cb}_{cc}"] = float(im_val)

    return result


def compute_invariants(
    tensors_dict: dict[str, np.ndarray | float],
    max_degree: int = 3,
    symmetry: Literal['O3', 'SO3', 'SO2'] = 'SO3',
    deduplicate_scalars: bool = True,
) -> dict[str, float]:
    """Compute SO(3), O(3), or SO(2) invariants from arbitrary Minkowski tensors.

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

    symmetry : {'O3', 'SO3', 'SO2'}, default='SO3'
        Symmetry group for invariants:
        - 'O3': Only true scalars (invariant under rotations AND reflections)
        - 'SO3': Include pseudo-scalars (flip sign under reflections)
        - 'SO2': Invariants under rotations about the z-axis only.
          Provides more invariants than SO(3) by treating z-components
          independently. Useful for objects near a wall or with a preferred axis.

    deduplicate_scalars : bool, default=True
        If True, remove degree-1 scalars that are linearly dependent on others.
        Specifically, Tr(w102)/3 is removed when w100 is present, and
        Tr(w202)/3 is removed when w200 is present.
        For symmetry='SO2', this applies only to the trace key, never to the
        `{name}_zz` key.

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

    For symmetry='SO3'/'O3', invariant labels follow these patterns:
    - Degree 1: tensor name (e.g., 'w000', 'w020')
    - Degree 2: 'dot_{v1}_{v2}', 'frob_{T1}_{T2}'
    - Degree 3: 'qf_{v1}_{T}_{v2}', 'ttr_{T1}_{T2}_{T3}',
                'det_{v1}_{v2}_{v3}', 'comm_{T1}_{T2}_{v}'

    For symmetry='SO2', invariant labels follow these patterns:
    - Degree 1: '{name}' (scalars/traces), '{name}_z' (v_z), '{name}_zz' (M_zz)
    - Degree 2: 'd1_{ci}_{cj}' (|m|=1 pairs), 'd2_{ci}_{cj}' (|m|=2 pairs)
    - Degree 3: 'tp_re_{ca}_{cb}_{cc}', 'tp_im_{ca}_{cb}_{cc}'

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
    >>> inv_so2 = compute_invariants(tensors, symmetry='SO2', max_degree=1)
    >>> 'w010_z' in inv_so2
    True
    >>> 'w020_zz' in inv_so2
    True
    """
    if not tensors_dict:
        return {}

    _VALID_SYMMETRIES = {'O3', 'SO3', 'SO2'}
    if symmetry not in _VALID_SYMMETRIES:
        raise ValueError(
            f"Invalid symmetry '{symmetry}'. Must be one of {sorted(_VALID_SYMMETRIES)}."
        )

    # SO(2): entirely separate decomposition and invariant pipeline
    if symmetry == 'SO2':
        so2_dec = _decompose_so2(tensors_dict)
        result = {}
        result.update(_so2_degree1_scalars(so2_dec, deduplicate=deduplicate_scalars))
        if max_degree >= 2:
            result.update(_so2_doublet_inner_products(so2_dec))
        if max_degree >= 3:
            result.update(_so2_triple_products(so2_dec))
        return result

    # Decompose all tensors into SO(3) irreps
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
