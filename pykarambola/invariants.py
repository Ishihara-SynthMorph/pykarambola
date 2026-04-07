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
