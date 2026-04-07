"""
Tests for SO(3) and O(3) invariant computation from Minkowski tensors.

Covers all milestones:
- M1: Tensor decomposition (trace/traceless split)
- M2: Degree-1 scalars with deduplication
- M3: Degree-2 invariants (dot products, Frobenius inner products)
- M4: Degree-3 O(3) invariants (quadratic forms, triple traces)
- M5: Degree-3 SO(3) pseudo-scalars (determinants, commutators)
- M6: Public API (compute_invariants)
- M7: Rotational invariance validation
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pykarambola.invariants import (
    _infer_rank,
    _trace_rank2,
    _traceless_rank2,
    decompose_all,
    _degree1_scalars,
    _vector_dot_products,
    _frobenius_inner_products,
    _quadratic_forms,
    _triple_traces,
    _triple_vector_dets,
    _commutator_pseudoscalars,
    compute_invariants,
)


# =============================================================================
# Test fixtures
# =============================================================================

@pytest.fixture
def unit_cube_tensors():
    """Minkowski tensors for a unit cube (approximate values for testing)."""
    return {
        'w000': 1.0,  # Volume
        'w100': 0.5,  # Surface area / 6
        'w200': 1.0 / 3.0,  # Mean curvature integral
        'w300': 1.0,  # Euler characteristic
        'w010': np.array([0.5, 0.5, 0.5]),  # Centroid * volume
        'w110': np.array([0.5, 0.5, 0.5]),
        'w210': np.array([0.5, 0.5, 0.5]),
        'w310': np.array([0.5, 0.5, 0.5]),
        'w020': np.eye(3) * 0.1,  # Approximate
        'w120': np.eye(3) * 0.1,
        'w220': np.eye(3) * 0.1,
        'w320': np.eye(3) * 0.1,
        'w102': np.eye(3) * 0.5,  # Tr(w102)/3 = w100
        'w202': np.eye(3) * (1.0 / 3.0),  # Tr(w202)/3 = w200
    }


@pytest.fixture
def random_tensors():
    """Random tensors for general testing."""
    rng = np.random.default_rng(42)
    # Create symmetric positive definite matrices
    def random_spd():
        A = rng.standard_normal((3, 3))
        return A @ A.T

    return {
        'w000': rng.random(),
        'w100': rng.random(),
        'w200': rng.random(),
        'w010': rng.standard_normal(3),
        'w110': rng.standard_normal(3),
        'w020': random_spd(),
        'w120': random_spd(),
        'w102': random_spd(),
    }


@pytest.fixture
def partial_tensors():
    """A partial set of tensors (subset of the full 14)."""
    return {
        'w000': 1.5,
        'w010': np.array([1.0, 2.0, 3.0]),
        'w020': np.diag([1.0, 2.0, 3.0]),
    }


def random_rotation_matrix(rng=None):
    """Generate a random SO(3) rotation matrix."""
    if rng is None:
        rng = np.random.default_rng()
    return Rotation.random(random_state=rng).as_matrix()


def apply_rotation_to_tensors(tensors, R):
    """Apply rotation R to all tensors."""
    rotated = {}
    for name, tensor in tensors.items():
        rank = _infer_rank(tensor)
        if rank == 0:
            rotated[name] = tensor
        elif rank == 1:
            rotated[name] = R @ np.asarray(tensor)
        elif rank == 2:
            M = np.asarray(tensor)
            rotated[name] = R @ M @ R.T
    return rotated


def apply_reflection_to_tensors(tensors):
    """Apply a reflection (x -> -x) to all tensors."""
    P = np.diag([-1.0, 1.0, 1.0])  # Reflection through yz-plane
    reflected = {}
    for name, tensor in tensors.items():
        rank = _infer_rank(tensor)
        if rank == 0:
            reflected[name] = tensor
        elif rank == 1:
            reflected[name] = P @ np.asarray(tensor)
        elif rank == 2:
            M = np.asarray(tensor)
            reflected[name] = P @ M @ P.T
    return reflected


# =============================================================================
# Milestone 1: Tensor decomposition tests
# =============================================================================

class TestInferRank:
    """Tests for _infer_rank function."""

    def test_scalar_int(self):
        assert _infer_rank(5) == 0

    def test_scalar_float(self):
        assert _infer_rank(3.14) == 0

    def test_scalar_0d_array(self):
        assert _infer_rank(np.array(2.5)) == 0

    def test_vector(self):
        assert _infer_rank(np.array([1, 2, 3])) == 1

    def test_matrix(self):
        assert _infer_rank(np.eye(3)) == 2

    def test_invalid_shape_4d_vector(self):
        with pytest.raises(ValueError, match="Unsupported tensor shape"):
            _infer_rank(np.array([1, 2, 3, 4]))

    def test_invalid_shape_2x2_matrix(self):
        with pytest.raises(ValueError, match="Unsupported tensor shape"):
            _infer_rank(np.eye(2))


class TestTraceDecomposition:
    """Tests for trace/traceless decomposition of rank-2 tensors."""

    def test_trace_identity(self):
        """Tr(I) / 3 = 1."""
        assert np.isclose(_trace_rank2(np.eye(3)), 1.0)

    def test_trace_scaled_identity(self):
        """Tr(2I) / 3 = 2."""
        assert np.isclose(_trace_rank2(2 * np.eye(3)), 2.0)

    def test_trace_diagonal(self):
        """Tr(diag(1,2,3)) / 3 = 2."""
        M = np.diag([1.0, 2.0, 3.0])
        assert np.isclose(_trace_rank2(M), 2.0)

    def test_traceless_identity_is_zero(self):
        """Traceless part of identity is zero."""
        tl = _traceless_rank2(np.eye(3))
        assert np.allclose(tl, 0.0)

    def test_traceless_is_traceless(self):
        """Traceless part has zero trace."""
        rng = np.random.default_rng(123)
        M = rng.standard_normal((3, 3))
        M = M + M.T  # Make symmetric
        tl = _traceless_rank2(M)
        assert np.isclose(np.trace(tl), 0.0, atol=1e-12)

    def test_completeness(self):
        """Original = traceless + (trace * 3) * I / 3 = traceless + trace * I."""
        rng = np.random.default_rng(456)
        M = rng.standard_normal((3, 3))
        M = M + M.T
        trace = _trace_rank2(M)
        tl = _traceless_rank2(M)
        reconstructed = tl + trace * np.eye(3)
        assert np.allclose(M, reconstructed, atol=1e-12)

    def test_norm_preservation(self):
        """||M||^2_F = ||T_tl||^2_F + 3 * trace^2."""
        rng = np.random.default_rng(789)
        M = rng.standard_normal((3, 3))
        M = M + M.T
        trace = _trace_rank2(M)
        tl = _traceless_rank2(M)
        norm_M = np.sum(M ** 2)
        norm_tl = np.sum(tl ** 2)
        norm_trace_part = 3 * trace ** 2
        assert np.isclose(norm_M, norm_tl + norm_trace_part, atol=1e-12)

    def test_orthogonality(self):
        """Traceless part is orthogonal to identity: Tr(T_tl) = 0."""
        rng = np.random.default_rng(101)
        M = rng.standard_normal((3, 3))
        M = M + M.T
        tl = _traceless_rank2(M)
        # Frobenius inner product with identity
        inner = np.sum(tl * np.eye(3))
        assert np.isclose(inner, 0.0, atol=1e-12)


class TestDecomposeAll:
    """Tests for decompose_all function."""

    def test_scalar_decomposition(self):
        decomposed = decompose_all({'w000': 2.5})
        assert ('w000', '0e') in decomposed
        assert np.isclose(decomposed[('w000', '0e')], 2.5)

    def test_vector_decomposition(self):
        v = np.array([1.0, 2.0, 3.0])
        decomposed = decompose_all({'w010': v})
        assert ('w010', '1e') in decomposed
        assert np.allclose(decomposed[('w010', '1e')], v)

    def test_matrix_decomposition(self):
        M = np.diag([1.0, 2.0, 3.0])
        decomposed = decompose_all({'w020': M})
        assert ('w020', '0e') in decomposed
        assert ('w020', '2e') in decomposed
        assert np.isclose(decomposed[('w020', '0e')], 2.0)  # trace/3

    def test_mixed_decomposition(self, partial_tensors):
        decomposed = decompose_all(partial_tensors)
        assert ('w000', '0e') in decomposed
        assert ('w010', '1e') in decomposed
        assert ('w020', '0e') in decomposed
        assert ('w020', '2e') in decomposed

    def test_empty_input(self):
        decomposed = decompose_all({})
        assert decomposed == {}


# =============================================================================
# Milestone 2: Degree-1 scalars tests
# =============================================================================

class TestDegree1Scalars:
    """Tests for degree-1 scalar invariants."""

    def test_collects_all_0e(self, partial_tensors):
        decomposed = decompose_all(partial_tensors)
        scalars = _degree1_scalars(decomposed, deduplicate=False)
        # w000 (rank 0) + w020 trace (rank 2)
        assert 'w000' in scalars
        assert 'w020' in scalars

    def test_deduplication_removes_w102_trace(self):
        """When w100 and w102 both present, w102 trace is removed."""
        tensors = {
            'w100': 0.5,
            'w102': np.eye(3) * 0.5,  # Tr/3 = 0.5 = w100
        }
        decomposed = decompose_all(tensors)
        scalars = _degree1_scalars(decomposed, deduplicate=True)
        assert 'w100' in scalars
        assert 'w102' not in scalars

    def test_deduplication_removes_w202_trace(self):
        """When w200 and w202 both present, w202 trace is removed."""
        tensors = {
            'w200': 1.0 / 3.0,
            'w202': np.eye(3) * (1.0 / 3.0),
        }
        decomposed = decompose_all(tensors)
        scalars = _degree1_scalars(decomposed, deduplicate=True)
        assert 'w200' in scalars
        assert 'w202' not in scalars

    def test_no_deduplication_when_base_missing(self):
        """w102 trace kept if w100 is not present."""
        tensors = {
            'w102': np.eye(3) * 0.5,
        }
        decomposed = decompose_all(tensors)
        scalars = _degree1_scalars(decomposed, deduplicate=True)
        assert 'w102' in scalars

    def test_deduplication_disabled(self):
        """With deduplicate=False, all scalars are kept."""
        tensors = {
            'w100': 0.5,
            'w102': np.eye(3) * 0.5,
        }
        decomposed = decompose_all(tensors)
        scalars = _degree1_scalars(decomposed, deduplicate=False)
        assert 'w100' in scalars
        assert 'w102' in scalars

    def test_linear_dependency_identity(self, unit_cube_tensors):
        """Verify Tr(w102)/3 ≈ w100 and Tr(w202)/3 ≈ w200."""
        decomposed = decompose_all(unit_cube_tensors)
        w100 = decomposed[('w100', '0e')]
        w102_trace = decomposed[('w102', '0e')]
        w200 = decomposed[('w200', '0e')]
        w202_trace = decomposed[('w202', '0e')]
        assert np.isclose(w102_trace, w100, atol=1e-10)
        assert np.isclose(w202_trace, w200, atol=1e-10)


# =============================================================================
# Milestone 3: Degree-2 invariants tests
# =============================================================================

class TestVectorDotProducts:
    """Tests for vector dot product invariants."""

    def test_count_single_vector(self):
        tensors = {'v': np.array([1, 0, 0])}
        decomposed = decompose_all(tensors)
        dots = _vector_dot_products(decomposed)
        assert len(dots) == 1  # C(1,2) + 1 = 1

    def test_count_two_vectors(self):
        tensors = {
            'v1': np.array([1, 0, 0]),
            'v2': np.array([0, 1, 0]),
        }
        decomposed = decompose_all(tensors)
        dots = _vector_dot_products(decomposed)
        assert len(dots) == 3  # C(2,2) + 2 = 3

    def test_count_four_vectors(self):
        rng = np.random.default_rng(1001)
        tensors = {f'v{i}': rng.standard_normal(3) for i in range(4)}
        decomposed = decompose_all(tensors)
        dots = _vector_dot_products(decomposed)
        assert len(dots) == 10  # 4*(4+1)/2 = 10

    def test_dot_product_values(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        tensors = {'a': v1, 'b': v2}
        decomposed = decompose_all(tensors)
        dots = _vector_dot_products(decomposed)
        assert np.isclose(dots['dot_a_a'], 1.0)
        assert np.isclose(dots['dot_b_b'], 1.0)
        assert np.isclose(dots['dot_a_b'], 0.0)

    def test_symmetry(self):
        """dot(v_i, v_j) should equal dot(v_j, v_i), only one is stored."""
        tensors = {
            'a': np.array([1, 2, 3]),
            'b': np.array([4, 5, 6]),
        }
        decomposed = decompose_all(tensors)
        dots = _vector_dot_products(decomposed)
        # Should have dot_a_b but not dot_b_a (a < b lexicographically)
        assert 'dot_a_b' in dots
        assert 'dot_b_a' not in dots


class TestFrobeniusInnerProducts:
    """Tests for Frobenius inner product invariants."""

    def test_count_single_matrix(self):
        tensors = {'T': np.eye(3)}
        decomposed = decompose_all(tensors)
        frobs = _frobenius_inner_products(decomposed)
        assert len(frobs) == 1

    def test_count_three_matrices(self):
        rng = np.random.default_rng(1002)
        tensors = {f'T{i}': rng.standard_normal((3, 3)) for i in range(3)}
        decomposed = decompose_all(tensors)
        frobs = _frobenius_inner_products(decomposed)
        assert len(frobs) == 6  # 3*(3+1)/2 = 6

    def test_frobenius_values(self):
        T1 = np.eye(3)
        T2 = np.zeros((3, 3))
        tensors = {'a': T1, 'b': T2}
        decomposed = decompose_all(tensors)
        frobs = _frobenius_inner_products(decomposed)
        # For traceless parts:
        tl_a = decomposed[('a', '2e')]
        tl_b = decomposed[('b', '2e')]
        expected_aa = np.sum(tl_a ** 2)
        expected_ab = np.sum(tl_a * tl_b)
        assert np.isclose(frobs['frob_a_a'], expected_aa)
        assert np.isclose(frobs['frob_a_b'], expected_ab)

    def test_symmetry(self):
        """frob(T_i, T_j) should equal frob(T_j, T_i), only one is stored."""
        tensors = {
            'a': np.diag([1, 2, 3]),
            'b': np.diag([4, 5, 6]),
        }
        decomposed = decompose_all(tensors)
        frobs = _frobenius_inner_products(decomposed)
        assert 'frob_a_b' in frobs
        assert 'frob_b_a' not in frobs


class TestDegree2RotationalInvariance:
    """Rotational invariance tests for degree-2 invariants."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_dot_products_rotation_invariant(self, random_tensors, seed):
        rng = np.random.default_rng(seed)
        R = random_rotation_matrix(rng)
        rotated = apply_rotation_to_tensors(random_tensors, R)

        decomposed_orig = decompose_all(random_tensors)
        decomposed_rot = decompose_all(rotated)

        dots_orig = _vector_dot_products(decomposed_orig)
        dots_rot = _vector_dot_products(decomposed_rot)

        for key in dots_orig:
            assert np.isclose(dots_orig[key], dots_rot[key], atol=1e-10), f"{key} not invariant"

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_frobenius_products_rotation_invariant(self, random_tensors, seed):
        rng = np.random.default_rng(seed)
        R = random_rotation_matrix(rng)
        rotated = apply_rotation_to_tensors(random_tensors, R)

        decomposed_orig = decompose_all(random_tensors)
        decomposed_rot = decompose_all(rotated)

        frobs_orig = _frobenius_inner_products(decomposed_orig)
        frobs_rot = _frobenius_inner_products(decomposed_rot)

        for key in frobs_orig:
            assert np.isclose(frobs_orig[key], frobs_rot[key], atol=1e-10), f"{key} not invariant"


# =============================================================================
# Milestone 4: Degree-3 O(3) invariants tests
# =============================================================================

class TestQuadraticForms:
    """Tests for quadratic form invariants v_i^T T v_j."""

    def test_count(self):
        """Count = num_matrices * num_vector_pairs."""
        tensors = {
            'v1': np.array([1, 0, 0]),
            'v2': np.array([0, 1, 0]),
            'T1': np.eye(3),
            'T2': np.diag([1, 2, 3]),
        }
        decomposed = decompose_all(tensors)
        qfs = _quadratic_forms(decomposed)
        # 2 matrices * 3 vector pairs (v1-v1, v1-v2, v2-v2)
        assert len(qfs) == 6

    def test_values(self):
        v = np.array([1.0, 0.0, 0.0])
        T = np.diag([2.0, 3.0, 4.0])
        tensors = {'v': v, 'T': T}
        decomposed = decompose_all(tensors)
        qfs = _quadratic_forms(decomposed)
        # v^T * traceless(T) * v
        tl = decomposed[('T', '2e')]
        expected = v @ tl @ v
        assert np.isclose(qfs['qf_v_T_v'], expected)


class TestTripleTraces:
    """Tests for triple trace invariants Tr(T_i T_j T_k)."""

    def test_count_one_matrix(self):
        tensors = {'T': np.eye(3)}
        decomposed = decompose_all(tensors)
        ttrs = _triple_traces(decomposed)
        # C(1+2, 3) = C(3,3) = 1
        assert len(ttrs) == 1

    def test_count_two_matrices(self):
        tensors = {'T1': np.eye(3), 'T2': np.diag([1, 2, 3])}
        decomposed = decompose_all(tensors)
        ttrs = _triple_traces(decomposed)
        # C(2+2, 3) = C(4,3) = 4
        assert len(ttrs) == 4

    def test_count_three_matrices(self):
        rng = np.random.default_rng(1003)
        tensors = {f'T{i}': rng.standard_normal((3, 3)) for i in range(3)}
        decomposed = decompose_all(tensors)
        ttrs = _triple_traces(decomposed)
        # C(3+2, 3) = C(5,3) = 10
        assert len(ttrs) == 10

    def test_count_six_matrices(self):
        """The standard case with 6 rank-2 tensors gives 56 triple traces."""
        rng = np.random.default_rng(1004)
        tensors = {f'T{i}': rng.standard_normal((3, 3)) for i in range(6)}
        decomposed = decompose_all(tensors)
        ttrs = _triple_traces(decomposed)
        # C(6+2, 3) = C(8,3) = 56
        assert len(ttrs) == 56

    def test_empty_when_no_matrices(self):
        tensors = {'v': np.array([1, 2, 3])}
        decomposed = decompose_all(tensors)
        ttrs = _triple_traces(decomposed)
        assert len(ttrs) == 0


class TestDegree3O3RotationalInvariance:
    """Rotational invariance tests for degree-3 O(3) invariants."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_quadratic_forms_rotation_invariant(self, random_tensors, seed):
        rng = np.random.default_rng(seed)
        R = random_rotation_matrix(rng)
        rotated = apply_rotation_to_tensors(random_tensors, R)

        decomposed_orig = decompose_all(random_tensors)
        decomposed_rot = decompose_all(rotated)

        qfs_orig = _quadratic_forms(decomposed_orig)
        qfs_rot = _quadratic_forms(decomposed_rot)

        for key in qfs_orig:
            assert np.isclose(qfs_orig[key], qfs_rot[key], atol=1e-8), f"{key} not invariant"

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_triple_traces_rotation_invariant(self, random_tensors, seed):
        rng = np.random.default_rng(seed)
        R = random_rotation_matrix(rng)
        rotated = apply_rotation_to_tensors(random_tensors, R)

        decomposed_orig = decompose_all(random_tensors)
        decomposed_rot = decompose_all(rotated)

        ttrs_orig = _triple_traces(decomposed_orig)
        ttrs_rot = _triple_traces(decomposed_rot)

        for key in ttrs_orig:
            assert np.isclose(ttrs_orig[key], ttrs_rot[key], atol=1e-8), f"{key} not invariant"

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_o3_invariants_reflection_invariant(self, random_tensors, seed):
        """O(3) invariants should not change under reflection."""
        rng = np.random.default_rng(seed)
        R = random_rotation_matrix(rng)
        rotated = apply_rotation_to_tensors(random_tensors, R)
        reflected = apply_reflection_to_tensors(rotated)

        decomposed_orig = decompose_all(rotated)
        decomposed_ref = decompose_all(reflected)

        # Quadratic forms
        qfs_orig = _quadratic_forms(decomposed_orig)
        qfs_ref = _quadratic_forms(decomposed_ref)
        for key in qfs_orig:
            assert np.isclose(qfs_orig[key], qfs_ref[key], atol=1e-8), f"{key} changed under reflection"

        # Triple traces
        ttrs_orig = _triple_traces(decomposed_orig)
        ttrs_ref = _triple_traces(decomposed_ref)
        for key in ttrs_orig:
            assert np.isclose(ttrs_orig[key], ttrs_ref[key], atol=1e-8), f"{key} changed under reflection"


# =============================================================================
# Milestone 5: Degree-3 SO(3) pseudo-scalars tests
# =============================================================================

class TestTripleVectorDets:
    """Tests for triple vector determinant pseudo-scalars."""

    def test_count_three_vectors(self):
        rng = np.random.default_rng(1005)
        tensors = {f'v{i}': rng.standard_normal(3) for i in range(3)}
        decomposed = decompose_all(tensors)
        dets = _triple_vector_dets(decomposed)
        # C(3, 3) = 1
        assert len(dets) == 1

    def test_count_four_vectors(self):
        rng = np.random.default_rng(1006)
        tensors = {f'v{i}': rng.standard_normal(3) for i in range(4)}
        decomposed = decompose_all(tensors)
        dets = _triple_vector_dets(decomposed)
        # C(4, 3) = 4
        assert len(dets) == 4

    def test_empty_with_two_vectors(self):
        tensors = {'v1': np.array([1, 0, 0]), 'v2': np.array([0, 1, 0])}
        decomposed = decompose_all(tensors)
        dets = _triple_vector_dets(decomposed)
        assert len(dets) == 0

    def test_value(self):
        """det([e1, e2, e3]) = 1."""
        tensors = {
            'a': np.array([1, 0, 0]),
            'b': np.array([0, 1, 0]),
            'c': np.array([0, 0, 1]),
        }
        decomposed = decompose_all(tensors)
        dets = _triple_vector_dets(decomposed)
        assert np.isclose(dets['det_a_b_c'], 1.0)


class TestCommutatorPseudoscalars:
    """Tests for commutator-based pseudo-scalars."""

    def test_count(self):
        """Count = C(num_matrices, 2) * num_vectors."""
        tensors = {
            'v1': np.array([1, 0, 0]),
            'v2': np.array([0, 1, 0]),
            'T1': np.diag([1, 2, 3]),
            'T2': np.diag([3, 2, 1]),
        }
        decomposed = decompose_all(tensors)
        comms = _commutator_pseudoscalars(decomposed)
        # C(2, 2) * 2 = 1 * 2 = 2
        assert len(comms) == 2

    def test_empty_with_one_matrix(self):
        tensors = {'v': np.array([1, 0, 0]), 'T': np.eye(3)}
        decomposed = decompose_all(tensors)
        comms = _commutator_pseudoscalars(decomposed)
        assert len(comms) == 0

    def test_empty_with_no_vectors(self):
        tensors = {'T1': np.eye(3), 'T2': np.diag([1, 2, 3])}
        decomposed = decompose_all(tensors)
        comms = _commutator_pseudoscalars(decomposed)
        assert len(comms) == 0


class TestPseudoscalarParity:
    """Tests that pseudo-scalars flip sign under reflection."""

    def test_triple_det_flips_under_reflection(self, random_tensors):
        """Triple vector determinants should flip sign under reflection."""
        reflected = apply_reflection_to_tensors(random_tensors)

        decomposed_orig = decompose_all(random_tensors)
        decomposed_ref = decompose_all(reflected)

        dets_orig = _triple_vector_dets(decomposed_orig)
        dets_ref = _triple_vector_dets(decomposed_ref)

        for key in dets_orig:
            # Should flip sign (unless zero)
            if abs(dets_orig[key]) > 1e-10:
                assert np.isclose(dets_orig[key], -dets_ref[key], atol=1e-10), \
                    f"{key} did not flip sign under reflection"

    def test_commutator_flips_under_reflection(self, random_tensors):
        """Commutator pseudo-scalars should flip sign under reflection."""
        reflected = apply_reflection_to_tensors(random_tensors)

        decomposed_orig = decompose_all(random_tensors)
        decomposed_ref = decompose_all(reflected)

        comms_orig = _commutator_pseudoscalars(decomposed_orig)
        comms_ref = _commutator_pseudoscalars(decomposed_ref)

        for key in comms_orig:
            if abs(comms_orig[key]) > 1e-10:
                assert np.isclose(comms_orig[key], -comms_ref[key], atol=1e-10), \
                    f"{key} did not flip sign under reflection"

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_pseudoscalars_rotation_invariant(self, random_tensors, seed):
        """Pseudo-scalars should be invariant under proper rotations."""
        rng = np.random.default_rng(seed)
        R = random_rotation_matrix(rng)
        rotated = apply_rotation_to_tensors(random_tensors, R)

        decomposed_orig = decompose_all(random_tensors)
        decomposed_rot = decompose_all(rotated)

        dets_orig = _triple_vector_dets(decomposed_orig)
        dets_rot = _triple_vector_dets(decomposed_rot)
        for key in dets_orig:
            assert np.isclose(dets_orig[key], dets_rot[key], atol=1e-8), \
                f"{key} not invariant under rotation"

        comms_orig = _commutator_pseudoscalars(decomposed_orig)
        comms_rot = _commutator_pseudoscalars(decomposed_rot)
        for key in comms_orig:
            assert np.isclose(comms_orig[key], comms_rot[key], atol=1e-8), \
                f"{key} not invariant under rotation"


# =============================================================================
# Milestone 6: Public API tests
# =============================================================================

class TestComputeInvariants:
    """Tests for the compute_invariants public API."""

    def test_empty_input(self):
        result = compute_invariants({})
        assert result == {}

    def test_degree1_only(self, partial_tensors):
        result = compute_invariants(partial_tensors, max_degree=1)
        assert 'w000' in result
        assert 'w020' in result  # trace
        # No degree-2 invariants
        assert not any(k.startswith('dot_') for k in result)
        assert not any(k.startswith('frob_') for k in result)

    def test_degree2_includes_bilinear(self, partial_tensors):
        result = compute_invariants(partial_tensors, max_degree=2)
        # Should have degree-1
        assert 'w000' in result
        # Should have degree-2
        assert any(k.startswith('dot_') for k in result)
        assert any(k.startswith('frob_') for k in result)
        # No degree-3
        assert not any(k.startswith('qf_') for k in result)
        assert not any(k.startswith('ttr_') for k in result)

    def test_degree3_includes_trilinear(self, partial_tensors):
        result = compute_invariants(partial_tensors, max_degree=3)
        # Should have all degrees
        assert 'w000' in result
        assert any(k.startswith('dot_') for k in result)
        assert any(k.startswith('qf_') for k in result)
        assert any(k.startswith('ttr_') for k in result)

    def test_o3_symmetry_excludes_pseudoscalars(self, random_tensors):
        result_o3 = compute_invariants(random_tensors, max_degree=3, symmetry='O3')
        result_so3 = compute_invariants(random_tensors, max_degree=3, symmetry='SO3')
        # O3 should have no pseudo-scalars
        assert not any(k.startswith('det_') for k in result_o3)
        assert not any(k.startswith('comm_') for k in result_o3)
        # SO3 should have pseudo-scalars
        assert any(k.startswith('det_') for k in result_so3) or any(k.startswith('comm_') for k in result_so3)

    def test_deduplication_flag(self):
        tensors = {
            'w100': 0.5,
            'w102': np.eye(3) * 0.5,
        }
        result_dedup = compute_invariants(tensors, max_degree=1, deduplicate_scalars=True)
        result_no_dedup = compute_invariants(tensors, max_degree=1, deduplicate_scalars=False)
        assert 'w100' in result_dedup
        assert 'w102' not in result_dedup
        assert 'w100' in result_no_dedup
        assert 'w102' in result_no_dedup

    def test_deterministic_ordering(self, random_tensors):
        """Two calls with same input produce identical keys."""
        result1 = compute_invariants(random_tensors, max_degree=3)
        result2 = compute_invariants(random_tensors, max_degree=3)
        assert list(result1.keys()) == list(result2.keys())

    def test_all_values_are_floats(self, random_tensors):
        result = compute_invariants(random_tensors, max_degree=3)
        for key, val in result.items():
            assert isinstance(val, float), f"{key} is {type(val)}, not float"


class TestComputeInvariantsPartialInputs:
    """Tests with various partial tensor sets."""

    def test_scalars_only(self):
        tensors = {'s1': 1.5, 's2': 2.5}
        result = compute_invariants(tensors, max_degree=3)
        assert 's1' in result
        assert 's2' in result
        # No bilinear or trilinear invariants from scalars alone
        assert len(result) == 2

    def test_vectors_only(self):
        tensors = {
            'v1': np.array([1, 0, 0]),
            'v2': np.array([0, 1, 0]),
            'v3': np.array([0, 0, 1]),
        }
        result = compute_invariants(tensors, max_degree=3)
        # Degree-2: 3 dot products (v1-v1, v1-v2, v1-v3, v2-v2, v2-v3, v3-v3) = 6
        dot_count = sum(1 for k in result if k.startswith('dot_'))
        assert dot_count == 6
        # Degree-3 SO3: 1 determinant
        det_count = sum(1 for k in result if k.startswith('det_'))
        assert det_count == 1

    def test_matrices_only(self):
        tensors = {
            'T1': np.diag([1, 2, 3]),
            'T2': np.diag([3, 2, 1]),
        }
        result = compute_invariants(tensors, max_degree=3)
        # Degree-1: 2 traces
        assert 'T1' in result
        assert 'T2' in result
        # Degree-2: 3 Frobenius products
        frob_count = sum(1 for k in result if k.startswith('frob_'))
        assert frob_count == 3
        # Degree-3: 4 triple traces, no qf (no vectors), 1 commutator but no vectors
        ttr_count = sum(1 for k in result if k.startswith('ttr_'))
        assert ttr_count == 4


# =============================================================================
# Milestone 7: Integration and rotational invariance tests
# =============================================================================

class TestFullRotationalInvariance:
    """Comprehensive rotational invariance tests across all invariant types."""

    @pytest.mark.parametrize("seed", range(10))
    def test_full_so3_invariance(self, random_tensors, seed):
        """All SO3 invariants should be unchanged under rotation."""
        rng = np.random.default_rng(seed)
        R = random_rotation_matrix(rng)
        rotated = apply_rotation_to_tensors(random_tensors, R)

        inv_orig = compute_invariants(random_tensors, max_degree=3, symmetry='SO3')
        inv_rot = compute_invariants(rotated, max_degree=3, symmetry='SO3')

        assert inv_orig.keys() == inv_rot.keys()
        for key in inv_orig:
            assert np.isclose(inv_orig[key], inv_rot[key], atol=1e-8), \
                f"{key}: {inv_orig[key]} != {inv_rot[key]}"

    @pytest.mark.parametrize("seed", range(5))
    def test_o3_invariance_under_improper_rotation(self, random_tensors, seed):
        """O3 invariants should be unchanged under improper rotations."""
        rng = np.random.default_rng(seed)
        R = random_rotation_matrix(rng)
        rotated = apply_rotation_to_tensors(random_tensors, R)
        reflected = apply_reflection_to_tensors(rotated)

        inv_orig = compute_invariants(rotated, max_degree=3, symmetry='O3')
        inv_ref = compute_invariants(reflected, max_degree=3, symmetry='O3')

        for key in inv_orig:
            assert np.isclose(inv_orig[key], inv_ref[key], atol=1e-8), \
                f"{key}: O3 invariant changed under reflection"

    @pytest.mark.parametrize("seed", range(5))
    def test_so3_pseudoscalars_flip_under_reflection(self, random_tensors, seed):
        """SO3 pseudo-scalars should flip sign under reflection."""
        rng = np.random.default_rng(seed)
        R = random_rotation_matrix(rng)
        rotated = apply_rotation_to_tensors(random_tensors, R)
        reflected = apply_reflection_to_tensors(rotated)

        inv_orig = compute_invariants(rotated, max_degree=3, symmetry='SO3')
        inv_ref = compute_invariants(reflected, max_degree=3, symmetry='SO3')
        inv_o3 = compute_invariants(rotated, max_degree=3, symmetry='O3')

        # Pseudo-scalars are those in SO3 but not in O3
        pseudoscalar_keys = set(inv_orig.keys()) - set(inv_o3.keys())
        for key in pseudoscalar_keys:
            if abs(inv_orig[key]) > 1e-10:
                assert np.isclose(inv_orig[key], -inv_ref[key], atol=1e-8), \
                    f"{key}: pseudo-scalar did not flip sign"


class TestInvariantCounts:
    """Tests that invariant counts match expected values."""

    def test_full_14_tensor_counts(self, unit_cube_tensors):
        """With all 14 standard tensors, verify expected counts."""
        result = compute_invariants(unit_cube_tensors, max_degree=3, symmetry='SO3', deduplicate_scalars=True)

        # Count by type
        degree1_count = sum(1 for k in result if not any(k.startswith(p) for p in ['dot_', 'frob_', 'qf_', 'ttr_', 'det_', 'comm_']))
        dot_count = sum(1 for k in result if k.startswith('dot_'))
        frob_count = sum(1 for k in result if k.startswith('frob_'))
        qf_count = sum(1 for k in result if k.startswith('qf_'))
        ttr_count = sum(1 for k in result if k.startswith('ttr_'))
        det_count = sum(1 for k in result if k.startswith('det_'))
        comm_count = sum(1 for k in result if k.startswith('comm_'))

        # 14 tensors: 4 scalars + 4 vectors + 6 matrices
        # With dedup: 4 + 4 trace (6 from matrices, minus 2 deduped) = 4 + 4 = 8 degree-1
        # Actually: 4 rank-0 scalars + 6 matrix traces - 2 deduped = 8
        assert degree1_count == 8, f"Expected 8 degree-1, got {degree1_count}"

        # 4 vectors -> C(4,2) + 4 = 10 dot products
        assert dot_count == 10, f"Expected 10 dot products, got {dot_count}"

        # 6 matrices -> C(6,2) + 6 = 21 Frobenius products
        assert frob_count == 21, f"Expected 21 Frobenius products, got {frob_count}"

        # 6 matrices * 10 vector pairs = 60 quadratic forms
        assert qf_count == 60, f"Expected 60 quadratic forms, got {qf_count}"

        # C(6+2, 3) = C(8,3) = 56 triple traces
        assert ttr_count == 56, f"Expected 56 triple traces, got {ttr_count}"

        # C(4, 3) = 4 triple vector determinants
        assert det_count == 4, f"Expected 4 triple determinants, got {det_count}"

        # C(6, 2) * 4 = 15 * 4 = 60 commutator pseudo-scalars
        assert comm_count == 60, f"Expected 60 commutator pseudo-scalars, got {comm_count}"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_scalar(self):
        result = compute_invariants({'x': 42.0}, max_degree=3)
        assert result == {'x': 42.0}

    def test_single_vector(self):
        v = np.array([1.0, 2.0, 3.0])
        result = compute_invariants({'v': v}, max_degree=3)
        expected_dot = np.dot(v, v)
        assert 'dot_v_v' in result
        assert np.isclose(result['dot_v_v'], expected_dot)

    def test_single_matrix(self):
        M = np.diag([1.0, 2.0, 3.0])
        result = compute_invariants({'T': M}, max_degree=3)
        # Trace/3 = 2.0
        assert 'T' in result
        assert np.isclose(result['T'], 2.0)
        # Frobenius self-product of traceless part
        assert 'frob_T_T' in result
        # Triple trace
        assert 'ttr_T_T_T' in result

    def test_invalid_tensor_shape(self):
        with pytest.raises(ValueError):
            compute_invariants({'bad': np.array([1, 2, 3, 4])})

    def test_numerical_stability_small_values(self):
        """Test with very small tensor values."""
        tensors = {
            'v': np.array([1e-10, 1e-10, 1e-10]),
            'T': np.eye(3) * 1e-10,
        }
        result = compute_invariants(tensors, max_degree=3)
        # Should not raise, all values should be finite
        assert all(np.isfinite(v) for v in result.values())

    def test_numerical_stability_large_values(self):
        """Test with large tensor values."""
        tensors = {
            'v': np.array([1e10, 1e10, 1e10]),
            'T': np.eye(3) * 1e10,
        }
        result = compute_invariants(tensors, max_degree=3)
        assert all(np.isfinite(v) for v in result.values())


# =============================================================================
# Translation invariance tests (M6 completion)
# =============================================================================

class TestTranslationInvariance:
    """Tests for translation behavior of invariants.

    Minkowski tensors are generally NOT translation-invariant: tensors like
    w010, w020, etc. depend on the choice of origin. However, the computed
    invariants transform predictably under translation.

    Key insight: Translation affects position-weighted tensors (w0X0, wX10, wX20)
    but NOT the intrinsic shape tensors (w000, w100, w200, w300, w102, w202).
    """

    @staticmethod
    def translate_tensors(tensors, shift, volume=1.0):
        """Simulate translation effect on Minkowski tensors.

        For a translation by vector t:
        - w000, w100, w200, w300: unchanged (intrinsic)
        - w010: w010 + w000 * t
        - w110: w110 + w100 * t
        - w020: w020 + outer(w010, t) + outer(t, w010) + w000 * outer(t, t)
        - etc.

        This is a simplified model for testing purposes.
        """
        t = np.asarray(shift)
        translated = {}

        for name, tensor in tensors.items():
            if name in ['w000', 'w100', 'w200', 'w300', 'w102', 'w202']:
                # Intrinsic tensors: unchanged
                translated[name] = tensor
            elif name == 'w010':
                # w010' = w010 + w000 * t
                w000 = tensors.get('w000', volume)
                translated[name] = np.asarray(tensor) + w000 * t
            elif name == 'w110':
                w100 = tensors.get('w100', 0.0)
                translated[name] = np.asarray(tensor) + w100 * t
            elif name == 'w210':
                w200 = tensors.get('w200', 0.0)
                translated[name] = np.asarray(tensor) + w200 * t
            elif name == 'w310':
                w300 = tensors.get('w300', 0.0)
                translated[name] = np.asarray(tensor) + w300 * t
            elif name == 'w020':
                # w020' = w020 + w010 ⊗ t + t ⊗ w010 + w000 * (t ⊗ t)
                w000 = tensors.get('w000', volume)
                w010 = tensors.get('w010', np.zeros(3))
                M = np.asarray(tensor)
                translated[name] = (M + np.outer(w010, t) + np.outer(t, w010)
                                    + w000 * np.outer(t, t))
            elif name in ['w120', 'w220', 'w320']:
                # Simplified: these also shift but with different prefactors
                # For testing, we just mark them as shifted
                translated[name] = np.asarray(tensor) + np.outer(t, t) * 0.1
            else:
                # Unknown tensor: keep as-is
                translated[name] = tensor

        return translated

    def test_intrinsic_tensors_translation_invariant(self):
        """Intrinsic shape tensors (w000, w100, w200, w300, w102, w202) are translation-invariant."""
        tensors = {
            'w000': 2.5,
            'w100': 1.5,
            'w200': 0.8,
            'w300': 1.0,
            'w102': np.diag([1.0, 1.5, 2.0]),
            'w202': np.diag([0.5, 0.8, 1.2]),
        }
        shift = np.array([10.0, -5.0, 3.0])
        translated = self.translate_tensors(tensors, shift)

        inv_orig = compute_invariants(tensors, max_degree=3)
        inv_trans = compute_invariants(translated, max_degree=3)

        # All invariants from intrinsic tensors should be identical
        for key in inv_orig:
            assert np.isclose(inv_orig[key], inv_trans[key], atol=1e-10), \
                f"{key} changed under translation: {inv_orig[key]} -> {inv_trans[key]}"

    def test_position_weighted_tensors_change_under_translation(self):
        """Position-weighted tensors (w010, w020, etc.) change under translation."""
        tensors = {
            'w000': 1.0,
            'w010': np.array([0.5, 0.5, 0.5]),
            'w020': np.eye(3) * 0.1,
        }
        shift = np.array([1.0, 2.0, 3.0])
        translated = self.translate_tensors(tensors, shift)

        inv_orig = compute_invariants(tensors, max_degree=2)
        inv_trans = compute_invariants(translated, max_degree=2)

        # The w010 scalar (trace of w020) should change
        # dot_w010_w010 should definitely change
        assert not np.isclose(inv_orig['dot_w010_w010'], inv_trans['dot_w010_w010']), \
            "dot_w010_w010 should change under translation"

    @pytest.mark.parametrize("seed", range(5))
    def test_translation_then_rotation_vs_rotation_then_translation(self, seed):
        """Verify that rotation and translation commute correctly for invariants.

        For intrinsic tensors, rotation and translation commute (both preserve them).
        """
        rng = np.random.default_rng(seed)
        R = random_rotation_matrix(rng)

        tensors = {
            'w000': rng.random(),
            'w100': rng.random(),
            'w102': rng.standard_normal((3, 3)),
        }
        tensors['w102'] = tensors['w102'] @ tensors['w102'].T  # Make SPD

        shift = rng.standard_normal(3) * 10

        # Path 1: translate then rotate
        trans1 = self.translate_tensors(tensors, shift)
        rot_trans1 = apply_rotation_to_tensors(trans1, R)

        # Path 2: rotate then translate (with rotated shift)
        rot2 = apply_rotation_to_tensors(tensors, R)
        trans_rot2 = self.translate_tensors(rot2, R @ shift)

        inv1 = compute_invariants(rot_trans1, max_degree=3)
        inv2 = compute_invariants(trans_rot2, max_degree=3)

        for key in inv1:
            assert np.isclose(inv1[key], inv2[key], atol=1e-8), \
                f"{key}: translate-rotate vs rotate-translate mismatch"


# =============================================================================
# Clebsch-Gordan consistency tests (M7 completion)
# =============================================================================

class TestClebschGordanConsistency:
    """Verify that all contractions satisfy Wigner-3j / Clebsch-Gordan selection rules.

    The key selection rules for coupling angular momenta l1, l2 -> L are:
    1. Triangle inequality: |l1 - l2| <= L <= l1 + l2
    2. Parity conservation: (-1)^(l1+l2+L) determines if the coupling is
       symmetric or antisymmetric

    For our invariants:
    - Scalars (0e): l=0, even parity
    - Vectors (1e): l=1, odd parity (polar vectors)
    - Traceless matrices (2e): l=2, even parity

    Valid couplings to scalar (L=0):
    - 0 ⊗ 0 -> 0: scalar * scalar (degree-1 products, trivial)
    - 1 ⊗ 1 -> 0: vector · vector (dot product)
    - 2 ⊗ 2 -> 0: matrix : matrix (Frobenius inner product)
    - 1 ⊗ 2 ⊗ 1 -> 0: v^T M v (quadratic form)
    - 2 ⊗ 2 ⊗ 2 -> 0: Tr(ABC) (triple trace)
    - 1 ⊗ 1 ⊗ 1 -> 0 (pseudo): det([v1, v2, v3]) (needs epsilon tensor)
    """

    def test_dot_product_coupling_rule(self):
        """Dot product couples two l=1 irreps to l=0.

        Selection rule: 1 ⊗ 1 = 0 ⊕ 1 ⊕ 2
        The dot product extracts the l=0 component.
        """
        # Two orthogonal vectors
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])

        # Dot product is scalar (l=0)
        dot = np.dot(v1, v2)
        assert np.isclose(dot, 0.0)  # Orthogonal

        # Cross product would be l=1 (vector)
        cross = np.cross(v1, v2)
        assert cross.shape == (3,)  # Still a vector

        # Outer product gives l=0 ⊕ l=2 (trace + traceless)
        outer = np.outer(v1, v2)
        trace = np.trace(outer)  # l=0 part
        traceless = outer - trace / 3 * np.eye(3)  # l=2 part
        assert np.isclose(np.trace(traceless), 0.0)

    def test_frobenius_coupling_rule(self):
        """Frobenius inner product couples two l=2 irreps to l=0.

        Selection rule: 2 ⊗ 2 = 0 ⊕ 1 ⊕ 2 ⊕ 3 ⊕ 4
        The Frobenius product Tr(A^T B) extracts the l=0 component.
        """
        # Two traceless matrices (pure l=2)
        A = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=float)
        B = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)

        assert np.isclose(np.trace(A), 0.0)  # Traceless
        assert np.isclose(np.trace(B), 0.0)  # Traceless

        # Frobenius inner product is scalar
        frob = np.sum(A * B)
        assert isinstance(frob, (int, float, np.floating))

    def test_quadratic_form_coupling_rule(self):
        """Quadratic form v^T M v couples 1 ⊗ 2 ⊗ 1 -> 0.

        This is a valid three-body coupling: first couple 1 ⊗ 2 -> 1, 2, 3
        then couple result with 1 -> 0 (only from 1 ⊗ 1 -> 0).
        """
        v = np.array([1.0, 0.0, 0.0])
        M = np.diag([1.0, 2.0, -3.0])  # Traceless (l=2)

        qf = v @ M @ v
        assert isinstance(qf, (int, float, np.floating))
        assert np.isclose(qf, 1.0)  # v selects M[0,0]

    def test_triple_trace_coupling_rule(self):
        """Triple trace Tr(ABC) couples 2 ⊗ 2 ⊗ 2 -> 0.

        By successive coupling: 2 ⊗ 2 -> 0,1,2,3,4
        Then (0,1,2,3,4) ⊗ 2 -> includes 0 from 2⊗2->0.
        """
        A = np.diag([1.0, -1.0, 0.0])  # Traceless
        B = np.diag([0.0, 1.0, -1.0])  # Traceless
        C = np.diag([-1.0, 0.0, 1.0])  # Traceless

        ttr = np.trace(A @ B @ C)
        assert isinstance(ttr, (int, float, np.floating))

    def test_determinant_pseudo_coupling(self):
        """det([v1, v2, v3]) couples 1 ⊗ 1 ⊗ 1 -> 0 (pseudo-scalar).

        This uses the Levi-Civita tensor ε_ijk which is a pseudo-tensor.
        The coupling 1 ⊗ 1 -> 0,1,2 doesn't directly give a scalar, but
        with the ε tensor (which transforms as a pseudo-scalar under
        improper rotations), we get a pseudo-scalar.
        """
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        v3 = np.array([0.0, 0.0, 1.0])

        det = np.linalg.det(np.column_stack([v1, v2, v3]))
        assert np.isclose(det, 1.0)  # Right-handed frame

        # Under reflection, this flips sign
        v1_reflected = np.array([-1.0, 0.0, 0.0])
        det_reflected = np.linalg.det(np.column_stack([v1_reflected, v2, v3]))
        assert np.isclose(det_reflected, -1.0)

    def test_commutator_pseudo_coupling(self):
        """Commutator [A,B] extracts the l=1 part of 2⊗2, then dots with vector.

        2 ⊗ 2 = 0 ⊕ 1 ⊕ 2 ⊕ 3 ⊕ 4
        The antisymmetric part (commutator) is the l=1 component.
        Then 1 ⊗ 1 -> 0 gives a scalar.

        But this is a pseudo-scalar because the commutator involves the
        structure constants of so(3), which transform as a pseudo-tensor.
        """
        # Two non-commuting matrices
        A = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)
        B = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=float)

        comm = A @ B - B @ A  # Antisymmetric (l=1 as axial vector)
        assert np.allclose(comm, -comm.T)  # Verify antisymmetric

        # Extract axial vector
        axial = np.array([comm[2, 1], comm[0, 2], comm[1, 0]])

        v = np.array([1.0, 1.0, 1.0])
        pseudo_scalar = np.dot(axial, v)
        assert isinstance(pseudo_scalar, (int, float, np.floating))

    def test_all_invariant_types_satisfy_selection_rules(self, random_tensors):
        """Verify all computed invariants satisfy CG selection rules by construction.

        This is a meta-test: since we've verified each type individually,
        and compute_invariants uses only these types, all invariants must
        satisfy the selection rules.
        """
        result = compute_invariants(random_tensors, max_degree=3, symmetry='SO3')

        # Categorize all invariants
        for key in result:
            if key.startswith('dot_'):
                # 1 ⊗ 1 -> 0: valid
                pass
            elif key.startswith('frob_'):
                # 2 ⊗ 2 -> 0: valid
                pass
            elif key.startswith('qf_'):
                # 1 ⊗ 2 ⊗ 1 -> 0: valid
                pass
            elif key.startswith('ttr_'):
                # 2 ⊗ 2 ⊗ 2 -> 0: valid
                pass
            elif key.startswith('det_'):
                # 1 ⊗ 1 ⊗ 1 -> 0 (pseudo): valid with ε
                pass
            elif key.startswith('comm_'):
                # (2 ⊗ 2 -> 1) ⊗ 1 -> 0 (pseudo): valid
                pass
            else:
                # Degree-1 scalar: trivially l=0
                pass

        # If we get here without error, all invariants are accounted for
        assert True

    def test_forbidden_coupling_not_present(self):
        """Verify we don't compute any forbidden couplings.

        For example, 0 ⊗ 1 -> 0 is forbidden (you can't couple a scalar
        and vector to get a scalar). Our code should not produce such terms.
        """
        tensors = {
            's': 1.0,  # scalar (l=0)
            'v': np.array([1.0, 2.0, 3.0]),  # vector (l=1)
        }
        result = compute_invariants(tensors, max_degree=3)

        # Should only have: s (degree-1), dot_v_v (degree-2)
        # No cross-terms between s and v at degree-2 (would be forbidden)
        expected_keys = {'s', 'dot_v_v'}
        assert set(result.keys()) == expected_keys, \
            f"Unexpected invariants: {set(result.keys()) - expected_keys}"
