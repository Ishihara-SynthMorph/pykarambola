"""
SO(3)-invariant scalar construction from Minkowski tensors.

This module implements a pipeline that takes Minkowski tensors (ranks 0-2) as input
and returns a vector of basis invariants -- the irreducible SO(3)- or O(3)-invariant
scalars obtained by contracting quantities that are not individually invariant
(vectors and traceless matrices).

References
----------
- Geiger & Smidt (2022). "e3nn: Euclidean Neural Networks" -- SO(3) irrep theory
- Mecke & Schroeder-Turk. "Minkowski Tensor Shape Analysis" -- integral geometry
"""

import numpy as np
from itertools import combinations_with_replacement, combinations

# -----------------------------------------------------------------------------
# Tensor Registry
# -----------------------------------------------------------------------------

TENSOR_REGISTRY = {
    # Scalars (rank 0) -- irrep 0e (even parity)
    'w000': {'rank': 0, 'weight_type': 'position', 'irreps': ['0e']},
    'w100': {'rank': 0, 'weight_type': 'area', 'irreps': ['0e']},
    'w200': {'rank': 0, 'weight_type': 'curvature', 'irreps': ['0e']},
    'w300': {'rank': 0, 'weight_type': 'gaussian', 'irreps': ['0e']},
    # Vectors (rank 1) -- irrep 1o (odd parity)
    'w010': {'rank': 1, 'weight_type': 'position', 'irreps': ['1o']},
    'w110': {'rank': 1, 'weight_type': 'area', 'irreps': ['1o']},
    'w210': {'rank': 1, 'weight_type': 'curvature', 'irreps': ['1o']},
    'w310': {'rank': 1, 'weight_type': 'gaussian', 'irreps': ['1o']},
    # Rank-2 tensors -- irreps 0e (trace) + 2e (traceless)
    'w020': {'rank': 2, 'weight_type': 'position', 'irreps': ['0e', '2e']},
    'w120': {'rank': 2, 'weight_type': 'area', 'irreps': ['0e', '2e']},
    'w220': {'rank': 2, 'weight_type': 'curvature', 'irreps': ['0e', '2e']},
    'w320': {'rank': 2, 'weight_type': 'gaussian', 'irreps': ['0e', '2e']},
    'w102': {'rank': 2, 'weight_type': 'area_normal', 'irreps': ['0e', '2e']},
    'w202': {'rank': 2, 'weight_type': 'curvature_normal', 'irreps': ['0e', '2e']},
}

# Ordered lists for deterministic iteration
SCALARS = ['w000', 'w100', 'w200', 'w300']
VECTORS = ['w010', 'w110', 'w210', 'w310']
RANK2_TENSORS = ['w020', 'w120', 'w220', 'w320', 'w102', 'w202']

# Identity matrix for traceless decomposition
_I3 = np.eye(3, dtype=np.float64)

# Levi-Civita tensor (precomputed for pseudo-scalar contractions)
_LEVI_CIVITA = np.zeros((3, 3, 3), dtype=np.float64)
_LEVI_CIVITA[0, 1, 2] = _LEVI_CIVITA[1, 2, 0] = _LEVI_CIVITA[2, 0, 1] = 1.0
_LEVI_CIVITA[2, 1, 0] = _LEVI_CIVITA[1, 0, 2] = _LEVI_CIVITA[0, 2, 1] = -1.0


# -----------------------------------------------------------------------------
# Harmonic Decomposition (Clebsch-Gordan projection)
# -----------------------------------------------------------------------------

def trace_rank2(M):
    """Extract the 0e scalar component from a rank-2 tensor.

    Returns Tr(M) / 3, the normalized trace that represents the isotropic
    (scalar) part of the tensor in the SO(3) irreducible decomposition.

    Parameters
    ----------
    M : (3, 3) array_like
        Symmetric rank-2 tensor.

    Returns
    -------
    float
        The trace scalar t = Tr(M) / 3.

    Notes
    -----
    The factor of 3 normalizes so that M_isotropic = t * I has trace = 3t.
    """
    M = np.asarray(M, dtype=np.float64)
    return np.trace(M) / 3.0


def traceless_rank2(M):
    """Extract the 2e traceless component from a rank-2 tensor.

    Returns M - (Tr(M)/3) I, the deviatoric (traceless) part of the tensor
    that transforms under the l=2 (even parity) irreducible representation
    of SO(3).

    Parameters
    ----------
    M : (3, 3) array_like
        Symmetric rank-2 tensor.

    Returns
    -------
    (3, 3) ndarray
        The traceless tensor T_tl = M - (Tr(M)/3) I.

    Notes
    -----
    For a symmetric M, the traceless part is also symmetric.
    """
    M = np.asarray(M, dtype=np.float64)
    t = np.trace(M) / 3.0
    return M - t * _I3


def decompose_all(tensors_dict):
    """Decompose all Minkowski tensors into SO(3) irreducible components.

    Extracts irreducible representations from rank-0, rank-1, and rank-2
    tensors using the Clebsch-Gordan procedure:
    - Rank 0: scalar (0e) -- returned as-is
    - Rank 1: vector (1o) -- returned as-is
    - Rank 2: trace (0e) + traceless (2e) -- decomposed

    Parameters
    ----------
    tensors_dict : dict
        Output from minkowski_tensors(..., compute='standard').
        Expected keys: w000, w100, w200, w300 (scalars),
                       w010, w110, w210, w310 (vectors),
                       w020, w120, w220, w320, w102, w202 (rank-2 matrices).

    Returns
    -------
    dict
        Mapping from (tensor_name, irrep_label) tuples to irrep values:
        - ('w000', '0e') -> float (scalar value)
        - ('w020', '0e') -> float (trace / 3)
        - ('w020', '2e') -> (3, 3) ndarray (traceless matrix)
        - ('w010', '1o') -> (3,) ndarray (vector)
        - etc.

    Raises
    ------
    KeyError
        If a required tensor is missing from tensors_dict.

    Notes
    -----
    The decomposition satisfies (where t = Tr(M)/3):
    - Orthogonality: Tr(T_traceless) = 0
    - Completeness: T_traceless + t * I = M
    - Norm preservation: ||M||_F^2 = ||T_tl||_F^2 + 3 * t^2
    """
    decomposed = {}

    # Process scalars (rank 0) -- already 0e irrep
    for name in SCALARS:
        if name not in tensors_dict:
            raise KeyError(f"Missing required scalar tensor: {name}")
        decomposed[(name, '0e')] = float(tensors_dict[name])

    # Process vectors (rank 1) -- already 1o irrep
    for name in VECTORS:
        if name not in tensors_dict:
            raise KeyError(f"Missing required vector tensor: {name}")
        decomposed[(name, '1o')] = np.asarray(tensors_dict[name], dtype=np.float64)

    # Process rank-2 tensors -- decompose into 0e (trace) + 2e (traceless)
    for name in RANK2_TENSORS:
        if name not in tensors_dict:
            raise KeyError(f"Missing required rank-2 tensor: {name}")
        M = np.asarray(tensors_dict[name], dtype=np.float64)
        decomposed[(name, '0e')] = trace_rank2(M)
        decomposed[(name, '2e')] = traceless_rank2(M)

    return decomposed


# -----------------------------------------------------------------------------
# Degree-1 Invariants (Scalars)
# -----------------------------------------------------------------------------

# Rank-2 tensors whose traces are included as independent scalars.
# Tr(w102)/3 = w100 and Tr(w202)/3 = w200 exactly (by construction), so they
# are excluded to avoid redundancy (Open Q#1 resolution: Option A).
_RANK2_INDEPENDENT_TRACES = ['w020', 'w120', 'w220', 'w320']

# Human-readable labels for degree-1 scalars (deterministic order)
_DEGREE1_LABELS = [
    's0_w000',        # volume
    's1_w100',        # surface area
    's2_w200',        # mean curvature integral
    's3_w300',        # Gaussian curvature integral
    's4_tr_w020',     # Tr(w020)/3
    's5_tr_w120',     # Tr(w120)/3
    's6_tr_w220',     # Tr(w220)/3
    's7_tr_w320',     # Tr(w320)/3
]


def _degree1_scalars(decomposed):
    """Extract degree-1 basis invariants (scalars).

    Collects 8 independent scalars:
    - s0-s3: the 4 rank-0 Minkowski scalars (w000, w100, w200, w300)
    - s4-s7: the traces of w020, w120, w220, w320 (Tr(M)/3)

    Note: Tr(w102)/3 = w100 and Tr(w202)/3 = w200 exactly (the trace of n⊗n
    integrated over the surface equals the surface area), so these are excluded
    to avoid linear dependency.

    Parameters
    ----------
    decomposed : dict
        Output from decompose_all().

    Returns
    -------
    scalars : np.ndarray, shape (8,)
        The degree-1 basis invariants.
    labels : list of str
        Human-readable labels matching the scalar positions.
    """
    scalars = np.zeros(8, dtype=np.float64)

    # s0-s3: rank-0 scalars
    for i, name in enumerate(SCALARS):
        scalars[i] = decomposed[(name, '0e')]

    # s4-s7: traces of w020, w120, w220, w320 (not w102, w202)
    for i, name in enumerate(_RANK2_INDEPENDENT_TRACES):
        scalars[4 + i] = decomposed[(name, '0e')]

    return scalars, list(_DEGREE1_LABELS)


# -----------------------------------------------------------------------------
# Degree-2 Invariants (Dot Products & Frobenius Inner Products)
# -----------------------------------------------------------------------------

# Short aliases for vectors (v0-v3) and traceless tensors (T0-T5)
_VECTOR_ALIASES = {
    'v0': 'w010', 'v1': 'w110', 'v2': 'w210', 'v3': 'w310'
}
_TRACELESS_ALIASES = {
    'T0': 'w020', 'T1': 'w120', 'T2': 'w220', 'T3': 'w320', 'T4': 'w102', 'T5': 'w202'
}


def _vector_dot_products(decomposed):
    """Compute degree-2 vector dot product invariants v_i · v_j for i <= j.

    Parameters
    ----------
    decomposed : dict
        Output from decompose_all().

    Returns
    -------
    invariants : np.ndarray, shape (10,)
        The 10 dot product invariants.
    labels : list of str
        Human-readable labels.

    Notes
    -----
    The contraction v_i · v_j = sum_k v_i[k] * v_j[k] is SO(3)-invariant
    since ||Rv|| = ||v|| for any rotation R.
    """
    # Extract vectors in deterministic order
    vectors = []
    for alias in ['v0', 'v1', 'v2', 'v3']:
        name = _VECTOR_ALIASES[alias]
        vectors.append(decomposed[(name, '1o')])

    # Compute dot products for all i <= j (10 pairs)
    invariants = []
    labels = []
    for i in range(4):
        for j in range(i, 4):
            dot_val = np.dot(vectors[i], vectors[j])
            invariants.append(dot_val)
            labels.append(f'dot_v{i}_v{j}')

    return np.array(invariants, dtype=np.float64), labels


def _frobenius_inner_products(decomposed):
    """Compute degree-2 Frobenius inner product invariants Tr(T_i^T T_j) for i <= j.

    Parameters
    ----------
    decomposed : dict
        Output from decompose_all().

    Returns
    -------
    invariants : np.ndarray, shape (21,)
        The 21 Frobenius inner product invariants.
    labels : list of str
        Human-readable labels.

    Notes
    -----
    The Frobenius inner product Tr(A^T B) = sum_{ij} A_ij * B_ij is SO(3)-invariant.
    For symmetric traceless matrices T_i, T_j, this equals Tr(T_i T_j).
    """
    # Extract traceless tensors in deterministic order
    traceless = []
    for alias in ['T0', 'T1', 'T2', 'T3', 'T4', 'T5']:
        name = _TRACELESS_ALIASES[alias]
        traceless.append(decomposed[(name, '2e')])

    # Compute Frobenius inner products for all i <= j (21 pairs)
    invariants = []
    labels = []
    for i in range(6):
        for j in range(i, 6):
            # Frobenius inner product: Tr(T_i^T @ T_j) = sum of element-wise products
            frob_val = np.einsum('ij,ij->', traceless[i], traceless[j])
            invariants.append(frob_val)
            labels.append(f'frob_T{i}_T{j}')

    return np.array(invariants, dtype=np.float64), labels


def _degree2_contractions(decomposed):
    """Compute all degree-2 basis invariants.

    Combines vector dot products (10) and Frobenius inner products (21)
    for a total of 31 degree-2 invariants.

    Parameters
    ----------
    decomposed : dict
        Output from decompose_all().

    Returns
    -------
    invariants : np.ndarray, shape (31,)
        The degree-2 basis invariants.
    labels : list of str
        Human-readable labels matching the invariant positions.
    """
    dot_inv, dot_labels = _vector_dot_products(decomposed)
    frob_inv, frob_labels = _frobenius_inner_products(decomposed)

    invariants = np.concatenate([dot_inv, frob_inv])
    labels = dot_labels + frob_labels

    return invariants, labels


# -----------------------------------------------------------------------------
# Degree-3 O(3) Invariants (Quadratic Forms & Triple Traces)
# -----------------------------------------------------------------------------

def _quadratic_forms(decomposed):
    """Compute degree-3 quadratic form invariants v_i^T T_k v_j for i <= j.

    Parameters
    ----------
    decomposed : dict
        Output from decompose_all().

    Returns
    -------
    invariants : np.ndarray, shape (60,)
        The 60 quadratic form invariants.
    labels : list of str
        Human-readable labels.

    Notes
    -----
    For each of the 6 traceless matrices T_k and 10 symmetric vector pairs
    (v_i, v_j) with i <= j, compute v_i^T T_k v_j.

    Since T_k is symmetric, v_i^T T_k v_j = v_j^T T_k v_i, so the i <= j
    constraint avoids double-counting.

    This contraction is O(3)-invariant (invariant under rotations and reflections).
    """
    # Extract vectors and traceless tensors
    vectors = [decomposed[(_VECTOR_ALIASES[f'v{i}'], '1o')] for i in range(4)]
    traceless = [decomposed[(_TRACELESS_ALIASES[f'T{k}'], '2e')] for k in range(6)]

    invariants = []
    labels = []

    # For each traceless matrix T_k
    for k in range(6):
        T_k = traceless[k]
        # For each vector pair (v_i, v_j) with i <= j
        for i in range(4):
            for j in range(i, 4):
                # v_i^T T_k v_j = einsum('i,ij,j->', vi, Tk, vj)
                qf_val = np.einsum('i,ij,j->', vectors[i], T_k, vectors[j])
                invariants.append(qf_val)
                labels.append(f'qf_v{i}_T{k}_v{j}')

    return np.array(invariants, dtype=np.float64), labels


def _triple_traces(decomposed):
    """Compute degree-3 triple matrix trace invariants Tr(T_i T_j T_k).

    Parameters
    ----------
    decomposed : dict
        Output from decompose_all().

    Returns
    -------
    invariants : np.ndarray, shape (56,)
        The 56 triple trace invariants.
    labels : list of str
        Human-readable labels.

    Notes
    -----
    The contraction Tr(T_i T_j T_k) is symmetric under:
    - Cyclic shifts: Tr(ABC) = Tr(BCA) = Tr(CAB)
    - Reversal: Tr(ABC) = Tr(CBA) for symmetric matrices

    These symmetries generate the full S_3 group (order 6), so the number of
    distinct invariants is C(n+2,3) = n(n+1)(n+2)/6 for n tensors.
    For n=6: C(8,3) = 56 exactly (Open Q#2 resolution).

    We enumerate all multisets of size 3 from {0,1,2,3,4,5} using
    combinations_with_replacement.
    """
    # Extract traceless tensors
    traceless = [decomposed[(_TRACELESS_ALIASES[f'T{k}'], '2e')] for k in range(6)]

    invariants = []
    labels = []

    # Enumerate all multisets {i,j,k} with i <= j <= k
    for i, j, k in combinations_with_replacement(range(6), 3):
        # Tr(T_i T_j T_k) = einsum('ij,jk,ki->', Ti, Tj, Tk)
        trace_val = np.einsum('ij,jk,ki->', traceless[i], traceless[j], traceless[k])
        invariants.append(trace_val)
        labels.append(f'ttr_T{i}_T{j}_T{k}')

    return np.array(invariants, dtype=np.float64), labels


def _degree3_o3_contractions(decomposed):
    """Compute all degree-3 O(3)-invariant basis contractions.

    Combines quadratic forms (60) and triple traces (56) for a total of 116
    degree-3 invariants that are invariant under both rotations and reflections.

    Parameters
    ----------
    decomposed : dict
        Output from decompose_all().

    Returns
    -------
    invariants : np.ndarray, shape (116,)
        The degree-3 O(3)-invariant basis.
    labels : list of str
        Human-readable labels matching the invariant positions.
    """
    qf_inv, qf_labels = _quadratic_forms(decomposed)
    ttr_inv, ttr_labels = _triple_traces(decomposed)

    invariants = np.concatenate([qf_inv, ttr_inv])
    labels = qf_labels + ttr_labels

    return invariants, labels


# -----------------------------------------------------------------------------
# Degree-3 SO(3)-Only Pseudo-Scalars
# -----------------------------------------------------------------------------

def _triple_vector_determinants(decomposed):
    """Compute SO(3)-only triple vector determinant pseudo-scalars.

    Parameters
    ----------
    decomposed : dict
        Output from decompose_all().

    Returns
    -------
    invariants : np.ndarray, shape (4,)
        The 4 triple vector determinant pseudo-scalars.
    labels : list of str
        Human-readable labels.

    Notes
    -----
    For i < j < k, det([v_i, v_j, v_k]) = v_i · (v_j × v_k) is a pseudo-scalar
    that is SO(3)-invariant but changes sign under reflection.

    With 4 vectors, there are C(4,3) = 4 distinct combinations.

    These pseudo-scalars detect chirality: mirror-image shapes will have
    opposite signs.
    """
    vectors = [decomposed[(_VECTOR_ALIASES[f'v{i}'], '1o')] for i in range(4)]

    invariants = []
    labels = []

    # All strictly increasing triples (i < j < k)
    for i, j, k in combinations(range(4), 3):
        # det([vi, vj, vk]) = vi · (vj × vk)
        det_val = np.linalg.det(np.column_stack([vectors[i], vectors[j], vectors[k]]))
        invariants.append(det_val)
        labels.append(f'det_v{i}_v{j}_v{k}')

    return np.array(invariants, dtype=np.float64), labels


def _commutator_pseudoscalars(decomposed):
    """Compute SO(3)-only commutator pseudo-scalars.

    Parameters
    ----------
    decomposed : dict
        Output from decompose_all().

    Returns
    -------
    invariants : np.ndarray, shape (60,)
        The 60 commutator pseudo-scalar invariants.
    labels : list of str
        Human-readable labels.

    Notes
    -----
    For symmetric traceless matrices T_a, T_b and vector v_k:
    ψ(T_a, T_b, v_k) = ε_{ijk} [T_a, T_b]_{ij} v_k

    where [T_a, T_b] = T_a T_b - T_b T_a is the matrix commutator.

    This equals v_k dotted with the axial vector of the antisymmetric
    commutator [T_a, T_b]. Since [T_a, T_a] = 0, only the 15 off-diagonal
    pairs (a < b) contribute.

    15 matrix pairs × 4 vectors = 60 pseudo-scalars.

    These are SO(3)-invariant but change sign under reflection (pseudo-scalars).
    """
    vectors = [decomposed[(_VECTOR_ALIASES[f'v{i}'], '1o')] for i in range(4)]
    traceless = [decomposed[(_TRACELESS_ALIASES[f'T{k}'], '2e')] for k in range(6)]

    invariants = []
    labels = []

    # For each off-diagonal matrix pair (a < b)
    for a in range(6):
        for b in range(a + 1, 6):
            # Commutator: [T_a, T_b] = T_a @ T_b - T_b @ T_a
            comm = traceless[a] @ traceless[b] - traceless[b] @ traceless[a]

            # Extract axial vector from antisymmetric matrix:
            # axial[0] = comm[1,2], axial[1] = comm[2,0], axial[2] = comm[0,1]
            axial = np.array([comm[1, 2], comm[2, 0], comm[0, 1]])

            # For each vector v_k
            for k in range(4):
                # ψ = axial · v_k
                psi = np.dot(axial, vectors[k])
                invariants.append(psi)
                labels.append(f'comm_T{a}_T{b}_v{k}')

    return np.array(invariants, dtype=np.float64), labels


def _degree3_so3_only_pseudoscalars(decomposed):
    """Compute all degree-3 SO(3)-only pseudo-scalar invariants.

    Combines triple vector determinants (4) and commutator pseudo-scalars (60)
    for a total of 64 pseudo-scalar invariants that are SO(3)-invariant but
    change sign under reflection.

    Parameters
    ----------
    decomposed : dict
        Output from decompose_all().

    Returns
    -------
    invariants : np.ndarray, shape (64,)
        The degree-3 SO(3)-only pseudo-scalars.
    labels : list of str
        Human-readable labels matching the invariant positions.

    Notes
    -----
    These pseudo-scalars are chirality-sensitive: they distinguish mirror-image
    shapes. Include them with symmetry='SO3' to detect handedness in biological
    structures such as left- and right-handed helices.
    """
    det_inv, det_labels = _triple_vector_determinants(decomposed)
    comm_inv, comm_labels = _commutator_pseudoscalars(decomposed)

    invariants = np.concatenate([det_inv, comm_inv])
    labels = det_labels + comm_labels

    return invariants, labels
