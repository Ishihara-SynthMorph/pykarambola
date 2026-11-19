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

from pykarambola.api import minkowski_tensors
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
    _decompose_so2,
    _so2_degree1_scalars,
    _so2_collect_doublets,
    _so2_doublet_inner_products,
    _so2_triple_products,
    _drop_non_tensor_keys,
    compute_invariants,
    filter_by_degree,
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
    """Random tensors for general testing (includes >=3 vectors so det_ is non-empty)."""
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
        'w210': rng.standard_normal(3),
        'w310': rng.standard_normal(3),
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

    @pytest.mark.parametrize("bad_val", [
        True,
        False,
        "3.14",
        "hello",
        None,
        {"w000": 1.0},
        1 + 2j,
        np.array([True, False, True]),
        np.array([1 + 0j, 2 + 0j, 3 + 0j]),
    ])
    def test_invalid_dtype_or_type(self, bad_val):
        with pytest.raises(ValueError, match="Unsupported tensor"):
            _infer_rank(bad_val)

    def test_numpy_scalar_types(self):
        assert _infer_rank(np.int64(3)) == 0
        assert _infer_rank(np.float32(1.5)) == 0


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
        assert isinstance(decomposed[('w020', '0e')], float)

    def test_mixed_decomposition(self, partial_tensors):
        decomposed = decompose_all(partial_tensors)
        assert ('w000', '0e') in decomposed
        assert ('w010', '1e') in decomposed
        assert ('w020', '0e') in decomposed
        assert ('w020', '2e') in decomposed

    def test_empty_input(self):
        decomposed = decompose_all({})
        assert decomposed == {}

    def test_decompose_all_drops_eigensystem_keys(self):
        d = {
            'w020': np.eye(3),
            'w020_eigvals': np.array([1.0, 2.0, 3.0]),
            'w020_eigvecs': np.eye(3),
        }
        with pytest.warns(UserWarning, match="decompose_all: ignoring non-tensor eigensystem keys"):
            decomposed = decompose_all(d)
        assert set(decomposed.keys()) == {('w020', '0e'), ('w020', '2e')}


class TestDropNonTensorKeys:
    """Tests for stripping derived eigensystem keys (eigvals/eigvecs)."""

    def test_drops_eigvals_and_eigvecs(self):
        d = {
            'w020': np.eye(3),
            'w020_eigvals': np.array([1.0, 2.0, 3.0]),
            'w020_eigvecs': np.eye(3),
        }
        with pytest.warns(UserWarning, match="eigensystem keys"):
            kept = _drop_non_tensor_keys(d)
        assert set(kept) == {'w020'}

    def test_no_warning_when_clean(self):
        d = {'w000': 1.0, 'w020': np.eye(3)}
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            kept = _drop_non_tensor_keys(d)
        assert set(kept) == {'w000', 'w020'}

    def test_compute_invariants_ignores_eigen_keys(self):
        clean = {'w000': 1.0, 'w010': np.array([1.0, 0.0, 0.0])}
        polluted = {**clean, 'w020_eigvals': np.array([5.0, 4.0, 3.0])}
        base = compute_invariants(clean, max_degree=2)
        with pytest.warns(UserWarning, match="eigensystem keys"):
            got = compute_invariants(polluted, max_degree=2)
        assert got == base

    def test_multilabel_int_keyed_dict_raises_clear_type_error(self):
        multilabel = {1: {'w000': 1.0}, 2: {'w000': 2.0}}
        with pytest.raises(TypeError, match="tensors_dict keys must be strings"):
            compute_invariants(multilabel)
        with pytest.raises(TypeError, match="tensors_dict keys must be strings"):
            decompose_all(multilabel)


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

        assert len(dets_orig) > 0
        tested = 0
        for key in dets_orig:
            if abs(dets_orig[key]) > 1e-10:
                tested += 1
                assert np.isclose(dets_orig[key], -dets_ref[key], atol=1e-10), \
                    f"{key} did not flip sign under reflection"
        assert tested > 0

    def test_commutator_flips_under_reflection(self, random_tensors):
        """Commutator pseudo-scalars should flip sign under reflection."""
        reflected = apply_reflection_to_tensors(random_tensors)

        decomposed_orig = decompose_all(random_tensors)
        decomposed_ref = decompose_all(reflected)

        comms_orig = _commutator_pseudoscalars(decomposed_orig)
        comms_ref = _commutator_pseudoscalars(decomposed_ref)

        assert len(comms_orig) > 0
        tested = 0
        for key in comms_orig:
            if abs(comms_orig[key]) > 1e-10:
                tested += 1
                assert np.isclose(comms_orig[key], -comms_ref[key], atol=1e-10), \
                    f"{key} did not flip sign under reflection"
        assert tested > 0

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
        assert len(dets_orig) > 0
        for key in dets_orig:
            assert np.isclose(dets_orig[key], dets_rot[key], atol=1e-8), \
                f"{key} not invariant under rotation"

        comms_orig = _commutator_pseudoscalars(decomposed_orig)
        comms_rot = _commutator_pseudoscalars(decomposed_rot)
        assert len(comms_orig) > 0
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

    @pytest.mark.parametrize("bad_degree", [0, -1, 4, True, False, 1.5, "2"])
    def test_invalid_max_degree_raises(self, partial_tensors, bad_degree):
        with pytest.raises(ValueError, match="max_degree must be 1, 2, or 3"):
            compute_invariants(partial_tensors, max_degree=bad_degree)
        # Also validated even when tensors_dict is empty
        with pytest.raises(ValueError, match="max_degree must be 1, 2, or 3"):
            compute_invariants({}, max_degree=bad_degree)

    def test_invalid_symmetry_on_empty_input_raises(self):
        with pytest.raises(ValueError, match="Invalid symmetry"):
            compute_invariants({}, symmetry='invalid')

    def test_numpy_integer_max_degree_accepted(self, partial_tensors):
        assert compute_invariants(partial_tensors, max_degree=np.int64(2)) == (
            compute_invariants(partial_tensors, max_degree=2)
        )

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
        assert not any(k.startswith('tr3_') for k in result)

    def test_degree3_includes_trilinear(self, partial_tensors):
        result = compute_invariants(partial_tensors, max_degree=3)
        # Should have all degrees
        assert 'w000' in result
        assert any(k.startswith('dot_') for k in result)
        assert any(k.startswith('qf_') for k in result)
        assert any(k.startswith('tr3_') for k in result)

    def test_o3_symmetry_excludes_pseudoscalars(self, random_tensors):
        result_o3 = compute_invariants(random_tensors, max_degree=3, symmetry='O3')
        result_so3 = compute_invariants(random_tensors, max_degree=3, symmetry='SO3')
        # O3 should have no pseudo-scalars
        assert not any(k.startswith('det_') for k in result_o3)
        assert not any(k.startswith('cpsc_') for k in result_o3)
        # SO3 should have both det_ and cpsc_ pseudo-scalars
        assert any(k.startswith('det_') for k in result_so3)
        assert any(k.startswith('cpsc_') for k in result_so3)

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
        # Degree-2: 6 dot products (v1-v1, v1-v2, v1-v3, v2-v2, v2-v3, v3-v3)
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
        ttr_count = sum(1 for k in result if k.startswith('tr3_'))
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
        assert any(k.startswith('det_') and abs(inv_orig[k]) > 1e-6 for k in inv_orig)
        assert any(k.startswith('cpsc_') and abs(inv_orig[k]) > 1e-6 for k in inv_orig)
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
        assert any(k.startswith('det_') and abs(inv_orig[k]) > 1e-6 for k in pseudoscalar_keys)
        assert any(k.startswith('cpsc_') and abs(inv_orig[k]) > 1e-6 for k in pseudoscalar_keys)
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
        degree1_count = sum(1 for k in result if not any(k.startswith(p) for p in ['dot_', 'frob_', 'qf_', 'tr3_', 'det_', 'cpsc_']))
        dot_count = sum(1 for k in result if k.startswith('dot_'))
        frob_count = sum(1 for k in result if k.startswith('frob_'))
        qf_count = sum(1 for k in result if k.startswith('qf_'))
        ttr_count = sum(1 for k in result if k.startswith('tr3_'))
        det_count = sum(1 for k in result if k.startswith('det_'))
        comm_count = sum(1 for k in result if k.startswith('cpsc_'))

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
        assert 'tr3_T_T_T' in result

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
        """Dot product couples two l=1 irreps to l=0 (1 ⊗ 1 -> 0)."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 2.0, 0.0])
        inv = compute_invariants({'a': v1, 'b': v2}, max_degree=2, symmetry='O3')
        assert set(inv.keys()) == {'dot_a_a', 'dot_a_b', 'dot_b_b'}
        assert inv['dot_a_a'] == pytest.approx(1.0)
        assert inv['dot_b_b'] == pytest.approx(4.0)
        assert inv['dot_a_b'] == pytest.approx(0.0)

    def test_frobenius_coupling_rule(self):
        """Frobenius inner product couples two pure l=2 irreps to l=0 (2 ⊗ 2 -> 0)
        and ignores the l=0 trace part."""
        A = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=float)
        B = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)
        # Adding isotropic multiples of I must not change frob_ (pure l=2 coupling)
        inv_pure = compute_invariants({'A': A, 'B': B}, max_degree=2, symmetry='O3')
        inv_shifted = compute_invariants(
            {'A': A + 5.0 * np.eye(3), 'B': B - 3.0 * np.eye(3)},
            max_degree=2, symmetry='O3',
        )
        assert inv_pure['frob_A_A'] == pytest.approx(2.0)
        assert inv_pure['frob_B_B'] == pytest.approx(2.0)
        assert inv_pure['frob_A_B'] == pytest.approx(0.0)
        for key in ('frob_A_A', 'frob_A_B', 'frob_B_B'):
            assert inv_shifted[key] == pytest.approx(inv_pure[key])

    def test_quadratic_form_coupling_rule(self):
        """Quadratic form v^T T v couples 1 ⊗ 2 ⊗ 1 -> 0 using only the traceless l=2 part."""
        v = np.array([1.0, 0.0, 0.0])
        M_tl = np.diag([1.0, 2.0, -3.0])  # traceless (pure l=2)
        inv_pure = compute_invariants({'M': M_tl, 'v': v}, max_degree=3, symmetry='O3')
        inv_shifted = compute_invariants(
            {'M': M_tl + 4.0 * np.eye(3), 'v': v}, max_degree=3, symmetry='O3',
        )
        assert inv_pure['qf_v_M_v'] == pytest.approx(1.0)
        assert inv_shifted['qf_v_M_v'] == pytest.approx(1.0)

    def test_triple_trace_coupling_rule(self):
        """Triple trace Tr(ABC) couples 2 ⊗ 2 ⊗ 2 -> 0 using only traceless l=2 parts."""
        A = np.diag([1.0, -1.0, 0.0])
        B = np.diag([0.0, 1.0, -1.0])
        C = np.diag([-1.0, 0.0, 1.0])
        inv = compute_invariants(
            {'A': A + 2.0 * np.eye(3), 'B': B - np.eye(3), 'C': C + 7.0 * np.eye(3)},
            max_degree=3, symmetry='O3',
        )
        assert inv['tr3_A_B_C'] == pytest.approx(float(np.trace(A @ B @ C)))
        assert inv['tr3_A_A_A'] == pytest.approx(float(np.trace(A @ A @ A)))

    def test_determinant_pseudo_coupling(self):
        """det([v1, v2, v3]) couples 1 ⊗ 1 ⊗ 1 -> 0o (included in SO3, excluded in O3)."""
        tensors = {
            'a': np.array([1.0, 0.0, 0.0]),
            'b': np.array([0.0, 1.0, 0.0]),
            'c': np.array([0.0, 0.0, 1.0]),
        }
        inv_so3 = compute_invariants(tensors, max_degree=3, symmetry='SO3')
        inv_o3 = compute_invariants(tensors, max_degree=3, symmetry='O3')
        assert inv_so3['det_a_b_c'] == pytest.approx(1.0)
        assert 'det_a_b_c' not in inv_o3

        reflected = apply_reflection_to_tensors(tensors)
        inv_so3_ref = compute_invariants(reflected, max_degree=3, symmetry='SO3')
        assert inv_so3_ref['det_a_b_c'] == pytest.approx(-1.0)

    def test_commutator_pseudo_coupling(self):
        """Commutator [A,B] extracts the antisymmetric l=1 part of 2 ⊗ 2 and dots with v.

        Verifies both the SO3/O3 parity split and the axial-vector sign convention
        a = (C_{12}, C_{20}, C_{01}) where C = A B - B A.
        """
        A = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)
        B = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=float)
        v = np.array([1.0, 2.0, 3.0])

        comm = A @ B - B @ A
        expected_axial = np.array([comm[1, 2], comm[2, 0], comm[0, 1]])
        expected_val = float(np.dot(expected_axial, v))
        assert abs(expected_val) > 0.5  # non-trivial

        tensors = {'A': A, 'B': B, 'v': v}
        inv_so3 = compute_invariants(tensors, max_degree=3, symmetry='SO3')
        inv_o3 = compute_invariants(tensors, max_degree=3, symmetry='O3')
        assert inv_so3['cpsc_A_B_v'] == pytest.approx(expected_val)
        assert 'cpsc_A_B_v' not in inv_o3

        reflected = apply_reflection_to_tensors(tensors)
        inv_so3_ref = compute_invariants(reflected, max_degree=3, symmetry='SO3')
        assert inv_so3_ref['cpsc_A_B_v'] == pytest.approx(-expected_val)

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

# =============================================================================
# TestSO2Invariants
# =============================================================================

class TestSO2Invariants:
    """Tests for SO(2) (z-rotation) invariant computation."""

    # ---- fixtures ----

    @pytest.fixture
    def simple_tensors(self):
        return {
            'w010': np.array([1.0, 2.0, 3.0]),
            'w020': np.diag([1.0, 2.0, 3.0]),
        }

    # ---- _decompose_so2 ----

    def test_decompose_so2_rank0(self):
        dec = _decompose_so2({'s': 3.7})
        assert ('s', 'sc') in dec
        assert dec[('s', 'sc')] == pytest.approx(3.7)

    def test_decompose_so2_rank1(self):
        v = np.array([1.0, 2.0, 3.0])
        dec = _decompose_so2({'v': v})
        assert dec[('v', 'z')] == pytest.approx(3.0)
        np.testing.assert_allclose(dec[('v', 'xy')], [1.0, 2.0])

    def test_decompose_so2_rank2(self):
        M = np.diag([1.0, 2.0, 3.0])
        dec = _decompose_so2({'M': M})
        tr_over3 = 2.0  # (1+2+3)/3
        assert dec[('M', 'tr')] == pytest.approx(tr_over3)
        assert dec[('M', 'tzz')] == pytest.approx(3.0 - tr_over3)  # M_zz - Tr/3
        np.testing.assert_allclose(dec[('M', 'xz')], [0.0, 0.0])
        np.testing.assert_allclose(dec[('M', 'm2')], [1.0 - 2.0, 0.0])  # [Mxx-Myy, 2Mxy]

    def test_decompose_so2_rank2_symmetrizes(self):
        """Non-symmetric input should be symmetrized."""
        M = np.array([[1., 3., 0.], [1., 2., 0.], [0., 0., 3.]])  # asymmetric
        dec = _decompose_so2({'M': M})
        # M_sym = [[1,2,0],[2,2,0],[0,0,3]]; M_xy=2, so m2[1] = 2*2 = 4
        np.testing.assert_allclose(dec[('M', 'm2')][1], 4.0)

    # ---- degree-1 scalars ----

    def test_so2_degree1_rank0(self):
        dec = _decompose_so2({'s': 7.0})
        scalars = _so2_degree1_scalars(dec)
        assert scalars['s'] == pytest.approx(7.0)

    def test_so2_degree1_rank1_z(self):
        v = np.array([1.0, 2.0, 5.0])
        dec = _decompose_so2({'v': v})
        scalars = _so2_degree1_scalars(dec)
        assert 'v_z' in scalars
        assert scalars['v_z'] == pytest.approx(5.0)
        assert 'v' not in scalars  # no rank-0 component from vectors

    def test_so2_degree1_rank2_trace_and_zz(self):
        M = np.diag([1.0, 2.0, 3.0])
        dec = _decompose_so2({'M': M})
        scalars = _so2_degree1_scalars(dec)
        assert 'M' in scalars
        assert scalars['M'] == pytest.approx(2.0)  # Tr/3 = 2
        assert 'M_zz' in scalars
        assert scalars['M_zz'] == pytest.approx(3.0)  # M[2,2]

    def test_so2_degree1_dedup_trace(self):
        """Tr(w102)/3 should be removed when w100 is present."""
        dec = _decompose_so2({'w100': 0.5, 'w102': np.diag([0.5, 0.5, 0.5])})
        scalars = _so2_degree1_scalars(dec, deduplicate=True)
        assert 'w100' in scalars
        assert 'w102' not in scalars   # deduped
        assert 'w102_zz' in scalars    # _zz always kept

    def test_so2_degree1_dedup_disabled(self):
        """With deduplicate=False both trace and base scalar appear."""
        dec = _decompose_so2({'w100': 0.5, 'w102': np.diag([0.5, 0.5, 0.5])})
        scalars = _so2_degree1_scalars(dec, deduplicate=False)
        assert 'w100' in scalars
        assert 'w102' in scalars
        assert 'w102_zz' in scalars

    def test_so2_degree1_zz_never_deduped(self):
        """_zz key should always be present even when trace is deduped."""
        dec = _decompose_so2({'w100': 0.5, 'w102': np.diag([1.5, 1.5, 1.5])})
        scalars_with = _so2_degree1_scalars(dec, deduplicate=True)
        scalars_without = _so2_degree1_scalars(dec, deduplicate=False)
        assert 'w102_zz' in scalars_with
        assert 'w102_zz' in scalars_without

    # ---- degree-2 doublet inner products ----

    def test_so2_m1_self_inner_product(self):
        v = np.array([3.0, 4.0, 0.0])
        dec = _decompose_so2({'v': v})
        inv = _so2_doublet_inner_products(dec)
        assert 'dot1_v_xy_v_xy' in inv
        assert inv['dot1_v_xy_v_xy'] == pytest.approx(25.0)  # 3²+4²

    def test_so2_m1_cross_type_pair(self):
        """Cross-type |m|=1 pair: rank-1 _xy and rank-2 _xz."""
        v = np.array([1.0, 0.0, 0.0])
        M = np.array([[0., 0., 1.], [0., 0., 0.], [1., 0., 0.]])  # M_xz=1
        dec = _decompose_so2({'v': v, 'M': M})
        inv = _so2_doublet_inner_products(dec)
        assert 'dot1_M_xz_v_xy' in inv or 'dot1_v_xy_M_xz' in inv
        # The actual key depends on sort order
        key = 'dot1_M_xz_v_xy' if 'dot1_M_xz_v_xy' in inv else 'dot1_v_xy_M_xz'
        assert inv[key] == pytest.approx(1.0)  # [1,0]·[1,0] = 1

    def test_so2_m2_self_inner_product(self):
        M = np.diag([2.0, -1.0, -1.0])
        dec = _decompose_so2({'M': M})
        inv = _so2_doublet_inner_products(dec)
        # m2 = [2-(-1), 2*0] = [3, 0]; |m2|² = 9
        assert 'dot2_M_m2_M_m2' in inv
        assert inv['dot2_M_m2_M_m2'] == pytest.approx(9.0)

    def test_so2_degree2_count(self):
        """With 1 rank-1 vector and 1 rank-2 matrix:
        |m|=1 doublets: v_xy + M_xz = 2 → 3 dot1_ dot pairs + 1 wed1_ wedge (i<j)
        |m|=2 doublets: M_m2 = 1 → 1 dot2_ pair + 0 wed2_ wedge (no distinct pair)
        Total: 5
        """
        dec = _decompose_so2({'v': np.array([1., 2., 3.]), 'M': np.eye(3)})
        inv = _so2_doublet_inner_products(dec)
        d1_keys = [k for k in inv if k.startswith('dot1_')]
        x1_keys = [k for k in inv if k.startswith('wed1_')]
        d2_keys = [k for k in inv if k.startswith('dot2_')]
        x2_keys = [k for k in inv if k.startswith('wed2_')]
        assert len(d1_keys) == 3
        assert len(x1_keys) == 1
        assert len(d2_keys) == 1
        assert len(x2_keys) == 0

    def test_so2_wedge_antisymmetric(self):
        """The wedge term flips sign when the two doublets are swapped, and the
        diagonal (same doublet) wedge is absent (would be zero)."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        inv = compute_invariants({'a': a, 'b': b}, symmetry='SO2', max_degree=2)
        # a_xy=[1,0], b_xy=[0,1] → wedge = 1*1 - 0*0 = 1
        assert inv['wed1_a_xy_b_xy'] == pytest.approx(1.0)
        assert 'wed1_a_xy_a_xy' not in inv  # diagonal wedge omitted

    # ---- degree-3 triple products ----

    def test_so2_triple_product_known_value_re(self):
        """Known value check for tp_re: v=[1,0,0], M=diag([2,-1,-1]).
        v_xy=[1,0], m2=[3,0]
        re_ab = 1*1 - 0*0 = 1, im_ab = 1*0 + 0*1 = 0
        tp_re = Re[conj(m2) * (v_xy*v_xy)] = 1*3 + 0*0 = 3
        tp_im = Im[conj(m2) * (v_xy*v_xy)] = 0*3 - 1*0 = 0
        """
        v = np.array([1.0, 0.0, 0.0])
        M = np.diag([2.0, -1.0, -1.0])
        inv = compute_invariants({'w010': v, 'w020': M}, symmetry='SO2', max_degree=3)
        assert inv['tp_re_w010_xy_w010_xy_w020_m2'] == pytest.approx(3.0)
        assert inv['tp_im_w010_xy_w010_xy_w020_m2'] == pytest.approx(0.0)

    def test_so2_triple_product_known_value_im(self):
        """Known value check for tp_im using inputs that make Im non-zero.
        v=[1,1,0], M with m2=[0,2] (M_xx=M_yy, M_xy=1).
        v_xy=[1,1], re_ab=1*1-1*1=0, im_ab=1*1+1*1=2
        tp_re = Re[conj(m2)*(v_xy*v_xy)] = 0*0 + 2*2 = 4
        tp_im = Im[conj(m2)*(v_xy*v_xy)] = 2*0 - 0*2 = 0... try different m2.

        Use v=[1,1,0], M s.t. m2=[1,0]:
        re_ab=0, im_ab=2
        tp_re = 0*1 + 2*0 = 0
        tp_im = 2*1 - 0*0 = 2
        """
        v = np.array([1.0, 1.0, 0.0])
        # M_xx - M_yy = 1, M_xy = 0  → m2 = [1, 0]
        M = np.array([[0.5, 0., 0.], [0., -0.5, 0.], [0., 0., 0.]])
        inv = compute_invariants({'w010': v, 'w020': M}, symmetry='SO2', max_degree=3)
        assert inv['tp_re_w010_xy_w010_xy_w020_m2'] == pytest.approx(0.0)
        assert inv['tp_im_w010_xy_w010_xy_w020_m2'] == pytest.approx(2.0)

    def test_so2_triple_product_re_im_both_present(self):
        """Both Re and Im keys should be generated for each triple."""
        v = np.array([1.0, 1.0, 0.0])
        M = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 0.]])  # off-diagonal
        dec = _decompose_so2({'v': v, 'M': M})
        tp = _so2_triple_products(dec)
        re_keys = [k for k in tp if k.startswith('tp_re_')]
        im_keys = [k for k in tp if k.startswith('tp_im_')]
        assert len(re_keys) == len(im_keys)
        assert len(re_keys) > 0

    def test_so2_triple_product_count(self):
        """With 1 rank-1 vector + 1 rank-2 matrix:
        |m|=1 doublets: 2 (v_xy, M_xz) → 2*3/2=3 pairs
        |m|=2 doublets: 1 (M_m2)
        → 3 * 1 * 2 (Re+Im) = 6 triple invariants
        """
        dec = _decompose_so2({'v': np.array([1., 2., 3.]), 'M': np.eye(3)})
        tp = _so2_triple_products(dec)
        assert len(tp) == 6

    # ---- invariance under z-rotation ----

    @pytest.mark.parametrize("theta", [15, 45, 90, 137, 270])
    def test_so2_invariance_under_z_rotation(self, theta, simple_tensors):
        """All SO(2) invariants must be unchanged by rotation about z."""
        inv = compute_invariants(simple_tensors, symmetry='SO2')
        R = Rotation.from_euler('z', theta, degrees=True).as_matrix()
        rotated = {
            k: (R @ v if np.asarray(v).ndim == 1 else R @ np.asarray(v) @ R.T)
            for k, v in simple_tensors.items()
        }
        inv_rot = compute_invariants(rotated, symmetry='SO2')
        for k in inv:
            assert abs(inv[k] - inv_rot[k]) < 1e-9, \
                f"z-rotation by {theta}° broke invariant '{k}': {inv[k]} vs {inv_rot[k]}"

    @pytest.mark.parametrize("seed", [0, 1, 7, 42])
    def test_so2_invariance_random_tensors(self, seed):
        """Random tensors: SO(2) invariants stable under z-rotation."""
        rng = np.random.default_rng(seed)
        M = rng.standard_normal((3, 3))
        M = (M + M.T) / 2
        tensors = {'v': rng.standard_normal(3), 'M': M}
        inv = compute_invariants(tensors, symmetry='SO2')
        theta = rng.uniform(0, 360)
        R = Rotation.from_euler('z', theta, degrees=True).as_matrix()
        rotated = {'v': R @ tensors['v'], 'M': R @ M @ R.T}
        inv_rot = compute_invariants(rotated, symmetry='SO2')
        for k in inv:
            assert abs(inv[k] - inv_rot[k]) < 1e-8, \
                f"seed={seed}, theta={theta:.1f}: '{k}' not invariant"

    # ---- NOT invariant under arbitrary rotation ----

    def test_so2_not_invariant_under_x_rotation(self, simple_tensors):
        """SO(2) invariants should NOT be invariant under rotation about x."""
        inv = compute_invariants(simple_tensors, symmetry='SO2')
        R = Rotation.from_euler('x', 45, degrees=True).as_matrix()
        rotated = {
            k: (R @ v if np.asarray(v).ndim == 1 else R @ np.asarray(v) @ R.T)
            for k, v in simple_tensors.items()
        }
        inv_rot = compute_invariants(rotated, symmetry='SO2')
        # w010_z should change since the vector is not aligned with z
        assert abs(inv['w010_z'] - inv_rot['w010_z']) > 1e-6, \
            "w010_z should change under x-rotation"

    # ---- invariant count for 14 standard tensors ----

    def test_so2_invariant_count_14_tensors(self):
        """Full 14-tensor set: expect 18 + 136 + 660 = 814 invariants."""
        rng = np.random.default_rng(0)

        def rand_sym(n=3):
            M = rng.standard_normal((n, n))
            return (M + M.T) / 2

        tensors = {
            # 4 rank-0 scalars
            'w000': 1.0, 'w100': 0.5, 'w200': 0.3, 'w300': 1.0,
            # 4 rank-1 vectors
            'w010': rng.standard_normal(3),
            'w110': rng.standard_normal(3),
            'w210': rng.standard_normal(3),
            'w310': rng.standard_normal(3),
            # 6 rank-2 matrices
            'w020': rand_sym(),
            'w120': rand_sym(),
            'w220': rand_sym(),
            'w320': rand_sym(),
            'w102': rand_sym(),
            'w202': rand_sym(),
        }

        inv = compute_invariants(tensors, symmetry='SO2', max_degree=3)

        # Degree-1: 4 scalars + 4 v_z + 4 trace (2 deduped: w102, w202) + 6 _zz = 18
        deg2_prefixes = ('dot1_', 'wed1_', 'dot2_', 'wed2_', 'tp_')
        deg1 = {k: v for k, v in inv.items() if not k.startswith(deg2_prefixes)}
        assert len(deg1) == 18, f"Expected 18 degree-1, got {len(deg1)}: {sorted(deg1)}"

        # Degree-2: 55 dot1_ + 45 wed1_ + 21 dot2_ + 15 wed2_ = 136
        deg2 = {k: v for k, v in inv.items() if k.startswith(('dot1_', 'wed1_', 'dot2_', 'wed2_'))}
        assert len(deg2) == 136, f"Expected 136 degree-2, got {len(deg2)}"

        # Degree-3: 660 triple products (Re + Im)
        deg3 = {k: v for k, v in inv.items() if k.startswith('tp_')}
        assert len(deg3) == 660, f"Expected 660 degree-3, got {len(deg3)}"

        assert len(inv) == 814, f"Expected 814 total, got {len(inv)}"

    # ---- API edge cases ----

    def test_so2_empty_input(self):
        assert compute_invariants({}, symmetry='SO2') == {}

    def test_so2_invalid_symmetry_raises(self):
        """Invalid symmetry strings should raise ValueError, not silently fall through."""
        tensors = {'v': np.array([1., 0., 0.])}
        with pytest.raises(ValueError, match="Invalid symmetry"):
            compute_invariants(tensors, symmetry='so2')
        with pytest.raises(ValueError, match="Invalid symmetry"):
            compute_invariants(tensors, symmetry='SO2 ')
        with pytest.raises(ValueError, match="Invalid symmetry"):
            compute_invariants(tensors, symmetry='xyz')

    def test_so2_degree1_key_collision_raises(self):
        """Degree-1 key collision between rank-0 name and rank-1 derived key raises."""
        tensors = {'v_z': 99.0, 'v': np.array([1., 2., 3.])}
        with pytest.raises(ValueError, match="key collision"):
            compute_invariants(tensors, symmetry='SO2', max_degree=1)

    def test_so2_scalar_only(self):
        inv = compute_invariants({'s': 3.0}, symmetry='SO2')
        assert inv == {'s': pytest.approx(3.0)}

    def test_so2_max_degree_1(self, simple_tensors):
        inv = compute_invariants(simple_tensors, symmetry='SO2', max_degree=1)
        assert all(not k.startswith(('dot1_', 'dot2_', 'tp_')) for k in inv)

    def test_so2_max_degree_2(self, simple_tensors):
        inv = compute_invariants(simple_tensors, symmetry='SO2', max_degree=2)
        assert not any(k.startswith('tp_') for k in inv)
        assert any(k.startswith(('dot1_', 'dot2_')) for k in inv)

# =============================================================================
# TestSO2InvariantsMeshIntegration
# =============================================================================

def _box_mesh_so2(a, b, c):
    """Axis-aligned box centered at the origin, 12 triangles, outward normals.

    Returns (verts, faces) numpy arrays. Duplicated locally to keep this module
    self-contained (avoids importing from another test file).
    """
    ha, hb, hc = a / 2.0, b / 2.0, c / 2.0
    verts = np.array([
        [-ha, -hb, -hc], [ ha, -hb, -hc], [ ha,  hb, -hc], [-ha,  hb, -hc],
        [-ha, -hb,  hc], [ ha, -hb,  hc], [ ha,  hb,  hc], [-ha,  hb,  hc],
    ], dtype=np.float64)
    faces = np.array([
        [0, 3, 2], [0, 2, 1],   # -z
        [4, 5, 6], [4, 6, 7],   # +z
        [0, 1, 5], [0, 5, 4],   # -y
        [2, 3, 7], [2, 7, 6],   # +y
        [0, 4, 7], [0, 7, 3],   # -x
        [1, 2, 6], [1, 6, 5],   # +x
    ], dtype=np.int64)
    return verts, faces


def _asymmetric_mesh():
    """Closed convex 12-triangle polyhedron with no symmetry planes, off-origin.

    All 14 Minkowski tensors and every SO(3)/O(3)/SO(2)/O(2) invariant family
    (including det_, cpsc_, wed1_, wed2_, tp_re_, tp_im_) are strictly non-zero.
    """
    verts = np.array([
        [-1.2, -0.8, -0.5], [ 1.7, -0.6, -0.4], [ 1.3,  1.1, -0.7], [-0.9,  0.9, -0.3],
        [-0.8, -1.1,  0.6], [ 1.4, -0.7,  1.3], [ 1.1,  1.4,  0.9], [-1.0,  0.7,  0.5],
    ], dtype=np.float64) + np.array([0.4, -0.3, 0.7])
    faces = np.array([
        [0, 3, 2], [0, 2, 1],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [2, 3, 7], [2, 7, 6],
        [0, 4, 7], [0, 7, 3],
        [1, 2, 6], [1, 6, 5],
    ], dtype=np.int64)
    return verts, faces


class TestSO2InvariantsMeshIntegration:
    """SO(2) invariants computed from mesh-derived Minkowski tensors.

    These tests exercise the full pipeline:
        mesh → minkowski_tensors() → compute_invariants(symmetry='SO2')

    All tests use an asymmetric box (3×2×1) so the invariants are non-trivial.
    Eigensystem keys (eigvals shape (3,), eigvecs shape (3,3)) are not spherical
    tensors; compute_invariants drops them on input with a warning, so passing
    the full minkowski_tensors() output is safe.
    """

    @pytest.fixture
    def box_tensors(self):
        """14 standard Minkowski tensors for the 3×2×1 box."""
        verts, faces = _box_mesh_so2(3.0, 2.0, 1.0)
        return minkowski_tensors(verts, faces, compute_eigensystems=False)

    # ---- basic pipeline ----

    def test_pipeline_runs(self, box_tensors):
        """minkowski_tensors() output is accepted without error."""
        inv = compute_invariants(box_tensors, symmetry='SO2')
        assert len(inv) > 0

    def test_degree1_keys_present(self, box_tensors):
        """Spot-check that expected degree-1 keys appear in the output."""
        inv = compute_invariants(box_tensors, symmetry='SO2', max_degree=1)
        # rank-0 scalars
        for name in ('w000', 'w100', 'w200', 'w300'):
            assert name in inv, f"Missing rank-0 key '{name}'"
        # rank-1 v_z
        for name in ('w010_z', 'w110_z', 'w210_z', 'w310_z'):
            assert name in inv, f"Missing v_z key '{name}'"
        # rank-2 _zz (always kept, even for deduped tensors)
        for name in ('w020_zz', 'w102_zz', 'w202_zz'):
            assert name in inv, f"Missing _zz key '{name}'"

    def test_invariant_count_814(self, box_tensors):
        """Full 14-tensor mesh output produces exactly 814 SO(2) invariants."""
        inv = compute_invariants(box_tensors, symmetry='SO2')
        assert len(inv) == 814

    # ---- deduplication with mesh data ----

    def test_dedup_w102_w100(self, box_tensors):
        """Tr(w102)/3 is proportional to w100 (both are area integrals), so the
        w102 trace key is deduped when w100 is present.  The _zz key is kept."""
        inv = compute_invariants(box_tensors, symmetry='SO2', max_degree=1)
        assert 'w100' in inv      # base scalar kept
        assert 'w102' not in inv  # trace of w102 is proportional to w100 → deduped
        assert 'w102_zz' in inv   # _zz = M_zz always kept

    def test_dedup_w202_w200(self, box_tensors):
        """Tr(w202)/3 is proportional to w200 (both are curvature integrals)."""
        inv = compute_invariants(box_tensors, symmetry='SO2', max_degree=1)
        assert 'w200' in inv
        assert 'w202' not in inv
        assert 'w202_zz' in inv

    def test_dedup_disabled_restores_both_scalars(self, box_tensors):
        """With deduplicate_scalars=False, w102 and w202 trace keys are kept."""
        inv = compute_invariants(box_tensors, symmetry='SO2', max_degree=1,
                                 deduplicate_scalars=False)
        assert 'w102' in inv
        assert 'w202' in inv

    # ---- z-rotation invariance on the actual mesh ----

    @pytest.mark.parametrize("theta", [30, 90, 180])
    def test_z_rotation_invariance(self, theta):
        """Rotating an asymmetric mesh about z produces identical SO(2) invariants
        across all non-zero families (dot1_, wed1_, dot2_, wed2_, tp_re_, tp_im_)."""
        verts, faces = _asymmetric_mesh()
        R = Rotation.from_euler('z', theta, degrees=True).as_matrix()

        tensors     = minkowski_tensors(verts,        faces, compute_eigensystems=False)
        tensors_rot = minkowski_tensors(verts @ R.T,  faces, compute_eigensystems=False)

        inv     = compute_invariants(tensors,     symmetry='SO2')
        inv_rot = compute_invariants(tensors_rot, symmetry='SO2')

        assert set(inv.keys()) == set(inv_rot.keys())
        for prefix in ('dot1_', 'wed1_', 'dot2_', 'wed2_', 'tp_re_', 'tp_im_'):
            vals = [abs(v) for k, v in inv.items() if k.startswith(prefix)]
            assert vals and max(vals) > 1e-6, f"Expected non-zero {prefix} invariants"
        for k in inv:
            assert inv[k] == pytest.approx(inv_rot[k], rel=1e-9, abs=1e-10), \
                f"z-rotation by {theta}° broke '{k}': {inv[k]:.6g} vs {inv_rot[k]:.6g}"

    # ---- non-invariance under off-axis rotation ----

    def test_not_invariant_under_x_rotation(self):
        """Rotating the mesh about x changes SO(2) invariants."""
        verts, faces = _box_mesh_so2(3.0, 2.0, 1.0)
        R = Rotation.from_euler('x', 45, degrees=True).as_matrix()

        tensors     = minkowski_tensors(verts,       faces, compute_eigensystems=False)
        tensors_rot = minkowski_tensors(verts @ R.T, faces, compute_eigensystems=False)

        inv     = compute_invariants(tensors,     symmetry='SO2')
        inv_rot = compute_invariants(tensors_rot, symmetry='SO2')

        # The box is asymmetric in z, so matrix _zz components change under x-rotation
        assert abs(inv['w020_zz'] - inv_rot['w020_zz']) > 1e-6, \
            "w020_zz should change under x-rotation of an asymmetric box"

    # ---- eigensystem-key pitfall ----

    def test_eigensystem_keys_dropped_with_warning(self):
        """Passing minkowski_tensors() output WITH eigensystems drops the derived
        eigvals/eigvecs keys (with a warning) and yields the same 814 invariants
        as the compute_eigensystems=False output."""
        verts, faces = _box_mesh_so2(3.0, 2.0, 1.0)
        tensors_with_eig = minkowski_tensors(verts, faces, compute_eigensystems=True)
        with pytest.warns(UserWarning, match="eigensystem keys"):
            inv = compute_invariants(tensors_with_eig, symmetry='SO2')
        assert len(inv) == 814

    def test_eigensystem_keys_yield_invariant_features(self):
        """After dropping eigensystem keys, the full-output result is rotation
        invariant under z-rotation (the footgun is fixed, not just documented)."""
        verts, faces = _box_mesh_so2(3.0, 2.0, 1.0)
        R = Rotation.from_euler('z', 37, degrees=True).as_matrix()
        with pytest.warns(UserWarning, match="eigensystem keys"):
            inv = compute_invariants(
                minkowski_tensors(verts, faces, compute_eigensystems=True),
                symmetry='SO2',
            )
            inv_rot = compute_invariants(
                minkowski_tensors(verts @ R.T, faces, compute_eigensystems=True),
                symmetry='SO2',
            )
        assert set(inv) == set(inv_rot)
        for k in inv:
            assert abs(inv[k] - inv_rot[k]) < 1e-10, f"z-rotation broke '{k}'"

# =============================================================================
# TestSO3InvariantsMeshIntegration
# =============================================================================

class TestSO3InvariantsMeshIntegration:
    """SO(3) and O(3) invariants computed from mesh-derived Minkowski tensors.

    These tests exercise the full pipeline:
        mesh → minkowski_tensors() → compute_invariants(symmetry='SO3'/'O3')

    As for SO(2), eigensystem keys (eigvals shape (3,), eigvecs shape (3,3)) are
    not spherical tensors; compute_invariants drops them on input with a warning,
    so passing the full minkowski_tensors() output is safe.

    Tolerance for rotation-invariance tests is 1e-5 (absolute).  The triple-trace
    invariants Tr(Ti @ Tj @ Tk) accumulate ~2.7e-7 floating-point error for the
    3×2×1 box under arbitrary rotations; 1e-5 provides a 40× safety margin.
    """

    @pytest.fixture
    def box_tensors(self):
        """14 standard Minkowski tensors for the 3×2×1 box."""
        verts, faces = _box_mesh_so2(3.0, 2.0, 1.0)
        return minkowski_tensors(verts, faces, compute_eigensystems=False)

    # ---- basic pipeline ----

    def test_pipeline_runs_so3(self, box_tensors):
        """minkowski_tensors() output is accepted for symmetry='SO3'."""
        inv = compute_invariants(box_tensors, symmetry='SO3')
        assert len(inv) > 0

    def test_pipeline_runs_o3(self, box_tensors):
        """minkowski_tensors() output is accepted for symmetry='O3'."""
        inv = compute_invariants(box_tensors, symmetry='O3')
        assert len(inv) > 0

    def test_degree1_keys_present(self, box_tensors):
        """Spot-check that expected degree-1 keys appear in the output."""
        inv = compute_invariants(box_tensors, symmetry='SO3', max_degree=1)
        for name in ('w000', 'w100', 'w200', 'w300',
                     'w020', 'w120', 'w220', 'w320'):
            assert name in inv, f"Missing degree-1 key '{name}'"

    # ---- invariant counts ----

    def test_invariant_count_so3(self, box_tensors):
        """14-tensor mesh output produces exactly 219 SO(3) invariants."""
        inv = compute_invariants(box_tensors, symmetry='SO3')
        assert len(inv) == 219

    def test_invariant_count_o3(self, box_tensors):
        """14-tensor mesh output produces exactly 155 O(3) invariants."""
        inv = compute_invariants(box_tensors, symmetry='O3')
        assert len(inv) == 155

    def test_so3_has_more_invariants_than_o3(self, box_tensors):
        """SO(3) includes pseudo-scalars (det_, cpsc_) absent from O(3)."""
        inv_so3 = compute_invariants(box_tensors, symmetry='SO3')
        inv_o3  = compute_invariants(box_tensors, symmetry='O3')
        assert len(inv_so3) > len(inv_o3)
        pseudo_keys = [k for k in inv_so3 if k not in inv_o3]
        assert all(k.startswith(('det_', 'cpsc_')) for k in pseudo_keys)

    # ---- deduplication with mesh data ----

    def test_dedup_w102_w100(self, box_tensors):
        """Tr(w102) = w100 for any mesh, so Tr(w102)/3 is proportional to w100
        and is removed by deduplicate_scalars=True."""
        inv = compute_invariants(box_tensors, symmetry='SO3', max_degree=1)
        assert 'w100' in inv
        assert 'w102' not in inv

    def test_dedup_w202_w200(self, box_tensors):
        """Tr(w202) = w200 for any mesh (curvature analogue of the area identity)."""
        inv = compute_invariants(box_tensors, symmetry='SO3', max_degree=1)
        assert 'w200' in inv
        assert 'w202' not in inv

    def test_dedup_disabled_restores_both_scalars(self, box_tensors):
        """With deduplicate_scalars=False, w102 and w202 trace keys are kept."""
        inv = compute_invariants(box_tensors, symmetry='SO3', max_degree=1,
                                 deduplicate_scalars=False)
        assert 'w102' in inv
        assert 'w202' in inv

    # ---- rotation invariance on the actual mesh ----

    @pytest.mark.parametrize("seed", [0, 1, 7, 42])
    def test_rotation_invariance_so3(self, seed):
        """Arbitrary rotation of an asymmetric mesh leaves SO(3) invariants unchanged,
        exercising non-zero values in every invariant family including det_ and cpsc_."""
        verts, faces = _asymmetric_mesh()
        R = Rotation.random(random_state=np.random.default_rng(seed)).as_matrix()

        tensors     = minkowski_tensors(verts,       faces, compute_eigensystems=False)
        tensors_rot = minkowski_tensors(verts @ R.T, faces, compute_eigensystems=False)

        inv     = compute_invariants(tensors,     symmetry='SO3')
        inv_rot = compute_invariants(tensors_rot, symmetry='SO3')

        assert set(inv.keys()) == set(inv_rot.keys())
        for prefix in ('dot_', 'frob_', 'qf_', 'tr3_', 'det_', 'cpsc_'):
            vals = [abs(v) for k, v in inv.items() if k.startswith(prefix)]
            assert vals and max(vals) > 1e-6, f"Expected non-zero {prefix} invariants"
        for k in inv:
            assert inv[k] == pytest.approx(inv_rot[k], rel=1e-6, abs=1e-5), \
                f"rotation broke '{k}': {inv[k]:.6g} vs {inv_rot[k]:.6g}"

    @pytest.mark.parametrize("seed", [0, 7])
    def test_rotation_invariance_o3(self, seed):
        """Arbitrary rotation or reflection of an asymmetric mesh leaves O(3) invariants unchanged."""
        verts, faces = _asymmetric_mesh()
        R = Rotation.random(random_state=np.random.default_rng(seed)).as_matrix()

        tensors     = minkowski_tensors(verts,       faces, compute_eigensystems=False)
        tensors_rot = minkowski_tensors(verts @ R.T, faces, compute_eigensystems=False)
        tensors_ref = minkowski_tensors(verts @ (-R).T, faces[:, ::-1], compute_eigensystems=False)

        inv     = compute_invariants(tensors,     symmetry='O3')
        inv_rot = compute_invariants(tensors_rot, symmetry='O3')
        inv_ref = compute_invariants(tensors_ref, symmetry='O3')

        for prefix in ('dot_', 'frob_', 'qf_', 'tr3_'):
            vals = [abs(v) for k, v in inv.items() if k.startswith(prefix)]
            assert vals and max(vals) > 1e-6, f"Expected non-zero {prefix} invariants"
        for k in inv:
            assert inv[k] == pytest.approx(inv_rot[k], rel=1e-6, abs=1e-5), \
                f"rotation broke '{k}': {inv[k]:.6g} vs {inv_rot[k]:.6g}"
            assert inv[k] == pytest.approx(inv_ref[k], rel=1e-6, abs=1e-5), \
                f"reflection broke '{k}': {inv[k]:.6g} vs {inv_ref[k]:.6g}"

    # ---- eigensystem-key pitfall ----

    def test_eigensystem_keys_dropped_with_warning(self):
        """Passing minkowski_tensors() output WITH eigensystems drops the derived
        eigvals/eigvecs keys (with a warning), yielding the same result as
        compute_eigensystems=False."""
        verts, faces = _box_mesh_so2(3.0, 2.0, 1.0)
        with pytest.warns(UserWarning, match="eigensystem keys"):
            inv_full = compute_invariants(
                minkowski_tensors(verts, faces, compute_eigensystems=True),
                symmetry='SO3',
            )
        inv_clean = compute_invariants(
            minkowski_tensors(verts, faces, compute_eigensystems=False),
            symmetry='SO3',
        )
        assert inv_full.keys() == inv_clean.keys()
        for k in inv_full:
            assert abs(inv_full[k] - inv_clean[k]) < 1e-12


# =============================================================================
# TestSO2SO3Consistency
# =============================================================================

class TestSO2SO3Consistency:
    """Verify that the independent SO(2) and SO(3) pipelines are mutually coherent.

    The two pipelines share no code: SO(3) goes through decompose_all() while
    SO(2) goes through _decompose_so2() directly.  These tests verify that the
    mathematical relationships between their outputs hold.

    Key identities tested:

    Degree-1:
        SO3[name]       == SO2[name]          for rank-0 scalars
        SO3[name]       == SO2[name]          for rank-2 traces (= Tr(M)/3)

    Degree-2 (vector dot product):
        SO3[dot_v_v] == SO2[v_z]**2 + SO2[dot1_v_xy_v_xy]

    Degree-2 (Frobenius of traceless rank-2 parts):
        SO3[frob_A_B] == (3/2)*tzz_A*tzz_B + SO2[dot2_Am2_Bm2]/2 + 2*SO2[dot1_Axz_Bxz]

        where  tzz_X = SO2[X_zz] - SO2[X]  (= M_zz - Tr(M)/3)
    """

    @pytest.fixture
    def tensors(self):
        return {
            'w000': 1.5,
            'w010': np.array([1.0, 2.0, 3.0]),
            'w020': np.array([[2.0, 1.0, 0.5],
                              [1.0, 3.0, 0.2],
                              [0.5, 0.2, 1.0]]),
            'w102': np.array([[1.0, 0.3, 0.1],
                              [0.3, 2.0, 0.4],
                              [0.1, 0.4, 0.5]]),
        }

    def test_degree1_rank0_match(self, tensors):
        so3 = compute_invariants(tensors, max_degree=1, symmetry='SO3',
                                 deduplicate_scalars=False)
        so2 = compute_invariants(tensors, max_degree=1, symmetry='SO2',
                                 deduplicate_scalars=False)
        assert so3['w000'] == pytest.approx(so2['w000'])

    def test_degree1_rank2_traces_match(self, tensors):
        so3 = compute_invariants(tensors, max_degree=1, symmetry='SO3',
                                 deduplicate_scalars=False)
        so2 = compute_invariants(tensors, max_degree=1, symmetry='SO2',
                                 deduplicate_scalars=False)
        for name in ('w020', 'w102'):
            assert so3[name] == pytest.approx(so2[name]), (
                f"trace mismatch for '{name}': SO3={so3[name]}, SO2={so2[name]}"
            )

    def test_degree2_dot_product_reconstruction(self, tensors):
        """SO(3) dot_v_v == SO(2) v_z^2 + dot(v_xy, v_xy)."""
        so3 = compute_invariants(tensors, max_degree=2, symmetry='SO3',
                                 deduplicate_scalars=False)
        so2 = compute_invariants(tensors, max_degree=2, symmetry='SO2',
                                 deduplicate_scalars=False)
        expected = so2['w010_z'] ** 2 + so2['dot1_w010_xy_w010_xy']
        assert so3['dot_w010_w010'] == pytest.approx(expected)

    @pytest.mark.parametrize('na,nb', [
        ('w020', 'w020'),
        ('w020', 'w102'),
        ('w102', 'w102'),
    ])
    def test_degree2_frobenius_reconstruction(self, tensors, na, nb):
        """SO(3) frob_A_B == (3/2)*tzz_A*tzz_B + dot2_Am2_Bm2/2 + 2*dot1_Axz_Bxz."""
        so3 = compute_invariants(tensors, max_degree=2, symmetry='SO3',
                                 deduplicate_scalars=False)
        so2 = compute_invariants(tensors, max_degree=2, symmetry='SO2',
                                 deduplicate_scalars=False)
        frob = so3[f'frob_{na}_{nb}']
        tzz_a = so2[f'{na}_zz'] - so2[na]
        tzz_b = so2[f'{nb}_zz'] - so2[nb]
        reconstructed = (
            1.5 * tzz_a * tzz_b
            + so2[f'dot2_{na}_m2_{nb}_m2'] / 2.0
            + 2.0 * so2[f'dot1_{na}_xz_{nb}_xz']
        )
        assert frob == pytest.approx(reconstructed), (
            f"Frobenius mismatch ({na},{nb}): SO3={frob:.6g}, "
            f"reconstructed={reconstructed:.6g}"
        )


# =============================================================================
# TestO2Invariants
# =============================================================================

class TestO2Invariants:
    """Tests for symmetry='O2' (C_inf_v: z-rotations + xy-plane reflections, preserving +z)."""

    @staticmethod
    def _transform_tensors(tensors, R):
        """Transform rank-0, rank-1, and rank-2 tensors by orthogonal matrix R."""
        out = {}
        for name, val in tensors.items():
            arr = np.asarray(val)
            if arr.ndim == 0:
                out[name] = float(arr)
            elif arr.shape == (3,):
                out[name] = R @ arr
            elif arr.shape == (3, 3):
                out[name] = R @ arr @ R.T
            else:
                raise ValueError(arr.shape)
        return out

    @pytest.fixture
    def standard_random_tensors(self):
        """14 standard Minkowski-shaped tensors with generic non-zero entries."""
        rng = np.random.default_rng(102)
        tensors = {
            'w000': float(rng.uniform(0.5, 2.0)),
            'w100': float(rng.uniform(0.5, 2.0)),
            'w200': float(rng.uniform(0.5, 2.0)),
            'w300': float(rng.uniform(0.5, 2.0)),
        }
        for name in ('w010', 'w110', 'w210', 'w310'):
            tensors[name] = rng.normal(size=3)
        for name in ('w020', 'w120', 'w220', 'w320', 'w102', 'w202'):
            A = rng.normal(size=(3, 3))
            tensors[name] = 0.5 * (A + A.T)
        return tensors

    # ---- subset & exact counts ----

    def test_o2_is_subset_of_so2_with_matching_values(self, standard_random_tensors):
        so2 = compute_invariants(standard_random_tensors, symmetry='SO2')
        o2 = compute_invariants(standard_random_tensors, symmetry='O2')
        assert set(o2.keys()) < set(so2.keys())
        for k, v in o2.items():
            assert v == pytest.approx(so2[k])

    def test_o2_invariant_counts_14_tensors(self, standard_random_tensors):
        """Standard 14 tensors produce 18 (deg 1) + 76 (deg 2) + 330 (deg 3) = 424 O(2) invariants."""
        o2 = compute_invariants(standard_random_tensors, symmetry='O2', max_degree=3)
        deg1 = filter_by_degree(o2, 1)
        deg2 = filter_by_degree(o2, 2)
        deg3 = filter_by_degree(o2, 3)

        assert len(deg1) == 18   # 4 rank-0 + 4 rank-1 v_z + 4 rank-2 trace (after dedup) + 6 rank-2 _zz
        assert len(deg2) == 76   # 55 dot1_ + 21 dot2_ (wed1_, wed2_ omitted)
        assert len(deg3) == 330  # 330 tp_re_ (tp_im_ omitted)
        assert len(o2) == 424

        assert sum(1 for k in deg2 if k.startswith('dot1_')) == 55
        assert sum(1 for k in deg2 if k.startswith('wed1_')) == 0
        assert sum(1 for k in deg2 if k.startswith('dot2_')) == 21
        assert sum(1 for k in deg2 if k.startswith('wed2_')) == 0
        assert sum(1 for k in deg3 if k.startswith('tp_re_')) == 330
        assert sum(1 for k in deg3 if k.startswith('tp_im_')) == 0

    def test_o2_omits_2d_pseudoscalars_and_preserves_vz(self, standard_random_tensors):
        o2 = compute_invariants(standard_random_tensors, symmetry='O2')
        for k in o2:
            assert not k.startswith(('wed1_', 'wed2_', 'tp_im_')), f"Unexpected pseudo-scalar key: {k}"
        for name in ('w010_z', 'w110_z', 'w210_z', 'w310_z'):
            assert name in o2

    # ---- symmetry properties ----

    @pytest.mark.parametrize('R_ref', [
        np.diag([1.0, -1.0, 1.0]),   # reflection across xz-plane (y -> -y)
        np.diag([-1.0, 1.0, 1.0]),   # reflection across yz-plane (x -> -x)
    ])
    def test_o2_invariant_under_xy_reflection_and_dropped_terms_flip_sign(
        self, standard_random_tensors, R_ref
    ):
        """O(2) terms are unchanged under xy-plane reflection, while every dropped
        SO(2) pseudo-scalar (wed1_, wed2_, tp_im_) strictly flips sign."""
        reflected = self._transform_tensors(standard_random_tensors, R_ref)

        so2_orig = compute_invariants(standard_random_tensors, symmetry='SO2')
        so2_ref = compute_invariants(reflected, symmetry='SO2')
        o2_orig = compute_invariants(standard_random_tensors, symmetry='O2')
        o2_ref = compute_invariants(reflected, symmetry='O2')

        # Every kept O(2) invariant is strictly unchanged
        for k in o2_orig:
            assert o2_ref[k] == pytest.approx(o2_orig[k], rel=1e-10, abs=1e-12), (
                f"O2 feature '{k}' not xy-reflection invariant: {o2_orig[k]} vs {o2_ref[k]}"
            )

        # Every dropped SO(2) feature is a genuine 2D pseudo-scalar (flips sign)
        dropped_keys = set(so2_orig.keys()) - set(o2_orig.keys())
        assert len(dropped_keys) == 814 - 424
        assert max(abs(so2_orig[k]) for k in dropped_keys) > 1e-3
        for k in dropped_keys:
            assert so2_ref[k] == pytest.approx(-so2_orig[k], rel=1e-10, abs=1e-12), (
                f"Dropped SO2 feature '{k}' did not flip sign: {so2_orig[k]} vs {so2_ref[k]}"
            )

    @pytest.mark.parametrize('theta', [0, 33, 90, 147, 270])
    def test_o2_invariant_under_z_rotation_and_vertical_reflection(self, theta, standard_random_tensors):
        R_rot = Rotation.from_euler('z', theta, degrees=True).as_matrix()
        R_combo = np.diag([1.0, -1.0, 1.0]) @ R_rot

        o2_orig = compute_invariants(standard_random_tensors, symmetry='O2')
        for R in (R_rot, R_combo):
            transformed = self._transform_tensors(standard_random_tensors, R)
            o2_new = compute_invariants(transformed, symmetry='O2')
            for k in o2_orig:
                assert o2_new[k] == pytest.approx(o2_orig[k], rel=1e-10, abs=1e-12)

    def test_o2_not_invariant_under_x_rotation_or_z_flip(self, standard_random_tensors):
        """O(2) (C_inf_v) preserves the oriented +z axis: neither x-rotation nor z -> -z
        leaves all O(2) features invariant."""
        o2_orig = compute_invariants(standard_random_tensors, symmetry='O2')

        R_x = Rotation.from_euler('x', 45, degrees=True).as_matrix()
        o2_rot = compute_invariants(self._transform_tensors(standard_random_tensors, R_x), symmetry='O2')
        assert max(abs(o2_orig[k] - o2_rot[k]) for k in o2_orig) > 1e-2

        R_zflip = np.diag([1.0, 1.0, -1.0])
        o2_zflip = compute_invariants(self._transform_tensors(standard_random_tensors, R_zflip), symmetry='O2')
        assert o2_zflip['w010_z'] == pytest.approx(-o2_orig['w010_z'])
        assert abs(o2_orig['w010_z']) > 1e-3

    def test_o2_reconstructs_o3_invariants(self, standard_random_tensors):
        """Because O(2) ⊂ O(3), every O(3) invariant must be expressible via O(2) invariants."""
        o3 = compute_invariants(standard_random_tensors, symmetry='O3', max_degree=2,
                                deduplicate_scalars=False)
        o2 = compute_invariants(standard_random_tensors, symmetry='O2', max_degree=2,
                                deduplicate_scalars=False)

        # dot(v, v) = v_z^2 + dot1(v_xy, v_xy)
        assert o3['dot_w010_w010'] == pytest.approx(
            o2['w010_z'] ** 2 + o2['dot1_w010_xy_w010_xy']
        )

        # Pure z-aligned vector must not vanish under O(2)
        pure_z = compute_invariants({'v': np.array([0.0, 0.0, 3.5])}, symmetry='O2')
        assert pure_z['v_z'] == pytest.approx(3.5)

        # Frobenius inner product of traceless rank-2 parts
        tzz_a = o2['w020_zz'] - o2['w020']
        tzz_b = o2['w120_zz'] - o2['w120']
        reconstructed = (
            1.5 * tzz_a * tzz_b
            + o2['dot2_w020_m2_w120_m2'] / 2.0
            + 2.0 * o2['dot1_w020_xz_w120_xz']
        )
        assert o3['frob_w020_w120'] == pytest.approx(reconstructed)

    def test_o2_custom_tensor_names(self):
        """Custom tensor names containing '_z', '_xy', '_xz' work cleanly in O(2)."""
        rng = np.random.default_rng(99)
        A = rng.normal(size=(3, 3))
        tensors = {
            'mass_z': 2.5,
            'v': rng.normal(size=3),
            'M_xy': 0.5 * (A + A.T),
        }
        o2_orig = compute_invariants(tensors, symmetry='O2')
        assert o2_orig['mass_z'] == pytest.approx(2.5)
        assert 'v_z' in o2_orig

        R = np.diag([1.0, -1.0, 1.0]) @ Rotation.from_euler('z', 40, degrees=True).as_matrix()
        o2_trans = compute_invariants(self._transform_tensors(tensors, R), symmetry='O2')
        for k in o2_orig:
            assert o2_trans[k] == pytest.approx(o2_orig[k], rel=1e-10, abs=1e-12)

    # ---- options and edge cases ----

    def test_o2_empty_and_max_degree_and_dedup(self, standard_random_tensors):
        assert compute_invariants({}, symmetry='O2') == {}

        o2_d1 = compute_invariants(standard_random_tensors, symmetry='O2', max_degree=1)
        o2_d2 = compute_invariants(standard_random_tensors, symmetry='O2', max_degree=2)
        o2_nodedup = compute_invariants(
            standard_random_tensors, symmetry='O2', max_degree=1, deduplicate_scalars=False
        )
        assert len(o2_d1) == 18
        assert len(o2_d2) == 18 + 76
        assert len(o2_nodedup) == 20  # restores w102 and w202 trace keys
        assert 'w102' in o2_nodedup and 'w202' in o2_nodedup


# =============================================================================
# TestO2InvariantsMeshIntegration
# =============================================================================

class TestO2InvariantsMeshIntegration:
    """End-to-end mesh -> minkowski_tensors -> compute_invariants(symmetry='O2') tests."""

    def test_mesh_o2_count_and_invariance(self):
        verts, faces = _asymmetric_mesh()
        tensors_orig = minkowski_tensors(verts, faces, compute_eigensystems=False)
        o2_orig = compute_invariants(tensors_orig, symmetry='O2')
        so2_orig = compute_invariants(tensors_orig, symmetry='SO2')
        assert len(o2_orig) == 424

        for prefix in ('dot1_', 'dot2_', 'tp_re_'):
            vals = [abs(v) for k, v in o2_orig.items() if k.startswith(prefix)]
            assert vals and max(vals) > 1e-6, f"Expected non-zero {prefix} invariants"
        assert max(abs(o2_orig[k]) for k in ('w010_z', 'w110_z', 'w210_z', 'w310_z')) > 1e-6

        # Reflect across xz / yz planes (flip face winding to preserve outward normals)
        # and rotate about z (no flip)
        R_yref = np.diag([1.0, -1.0, 1.0])
        R_xref = np.diag([-1.0, 1.0, 1.0])
        R_zrot = Rotation.from_euler('z', 53, degrees=True).as_matrix()
        for R, flip_faces in (
            (R_yref, True),
            (R_xref, True),
            (R_zrot, False),
            (R_yref @ R_zrot, True),
        ):
            verts_t = verts @ R.T
            faces_t = faces[:, ::-1] if flip_faces else faces
            tensors_t = minkowski_tensors(verts_t, faces_t, compute_eigensystems=False)
            o2_t = compute_invariants(tensors_t, symmetry='O2')
            for k in o2_orig:
                assert o2_t[k] == pytest.approx(o2_orig[k], rel=1e-6, abs=1e-6), (
                    f"Mesh O2 feature '{k}' differed: {o2_orig[k]} vs {o2_t[k]}"
                )

        # Verify dropped SO(2) pseudo-scalars genuinely flip sign under xy-plane reflection
        tensors_yref = minkowski_tensors(verts @ R_yref.T, faces[:, ::-1], compute_eigensystems=False)
        so2_yref = compute_invariants(tensors_yref, symmetry='SO2')
        dropped = set(so2_orig) - set(o2_orig)
        assert len(dropped) == 390
        assert max(abs(so2_orig[k]) for k in dropped) > 1e-6
        for k in dropped:
            assert so2_yref[k] == pytest.approx(-so2_orig[k], rel=1e-6, abs=1e-6), (
                f"Mesh dropped SO2 feature '{k}' did not flip sign: {so2_orig[k]} vs {so2_yref[k]}"
            )

    def test_mesh_o2_eigensystem_keys_dropped_with_warning(self):
        verts, faces = _asymmetric_mesh()
        tensors_full = minkowski_tensors(verts, faces, compute_eigensystems=True)
        tensors_clean = minkowski_tensors(verts, faces, compute_eigensystems=False)
        with pytest.warns(UserWarning, match="ignoring non-tensor eigensystem keys"):
            o2_full = compute_invariants(tensors_full, symmetry='O2')
        o2_clean = compute_invariants(tensors_clean, symmetry='O2')
        assert o2_full == o2_clean


# =============================================================================
# TestFilterByDegree
# =============================================================================

class TestFilterByDegree:
    """Tests for filter_by_degree() across all symmetry groups."""

    @pytest.fixture
    def sample_tensors(self):
        rng = np.random.default_rng(7)
        tensors = {'w000': 1.2, 'w100': 2.3, 'w200': 0.9, 'w300': 1.0}
        for name in ('w010', 'w110', 'w210', 'w310'):
            tensors[name] = rng.normal(size=3)
        for name in ('w020', 'w120', 'w220', 'w320', 'w102', 'w202'):
            A = rng.normal(size=(3, 3))
            tensors[name] = 0.5 * (A + A.T)
        return tensors

    def test_top_level_exports(self):
        import pykarambola
        assert pykarambola.compute_invariants is compute_invariants
        assert pykarambola.decompose_all is decompose_all
        assert pykarambola.filter_by_degree is filter_by_degree

    @pytest.mark.parametrize('symmetry,expected_counts', [
        ('O3',  (8, 31, 116)),
        ('SO3', (8, 31, 180)),
        ('SO2', (18, 136, 660)),
        ('O2',  (18, 76, 330)),
    ])
    def test_partition_and_max_degree_consistency(self, sample_tensors, symmetry, expected_counts):
        inv1 = compute_invariants(sample_tensors, max_degree=1, symmetry=symmetry)
        inv2 = compute_invariants(sample_tensors, max_degree=2, symmetry=symmetry)
        inv3 = compute_invariants(sample_tensors, max_degree=3, symmetry=symmetry)

        d1 = filter_by_degree(inv3, 1)
        d2 = filter_by_degree(inv3, 2)
        d3 = filter_by_degree(inv3, 3)

        # Counts match analytic expectations
        assert (len(d1), len(d2), len(d3)) == expected_counts

        # Degrees form a strict disjoint partition of inv3
        k1, k2, k3 = set(d1), set(d2), set(d3)
        assert k1.isdisjoint(k2) and k1.isdisjoint(k3) and k2.isdisjoint(k3)
        assert (k1 | k2 | k3) == set(inv3)

        # Filtering matches computing with lower max_degree directly
        assert d1 == inv1
        assert filter_by_degree(inv3, (1, 2)) == inv2
        assert filter_by_degree(inv3, [1, 2, 3]) == inv3

    def test_iterable_variants(self, sample_tensors):
        inv3 = compute_invariants(sample_tensors, max_degree=3, symmetry='SO3')
        d1 = filter_by_degree(inv3, 1)
        d3 = filter_by_degree(inv3, 3)

        # tuple, list, set, generator all accepted
        for spec in ((1, 3), [1, 3], {1, 3}, (d for d in (1, 3))):
            selected = filter_by_degree(inv3, spec)
            assert set(selected) == set(d1) | set(d3)
            for k, v in selected.items():
                assert v == inv3[k]

        # Empty iterable returns empty dict
        assert filter_by_degree(inv3, []) == {}

    def test_numpy_integer_degrees_accepted(self, sample_tensors):
        inv3 = compute_invariants(sample_tensors, max_degree=3, symmetry='SO3')
        assert filter_by_degree(inv3, np.int64(2)) == filter_by_degree(inv3, 2)
        assert (
            filter_by_degree(inv3, [np.int64(1), np.int32(3)])
            == filter_by_degree(inv3, (1, 3))
        )

    def test_empty_input_dict(self):
        assert filter_by_degree({}, 1) == {}
        assert filter_by_degree({}, (1, 2, 3)) == {}

    @pytest.mark.parametrize('bad_degree', [0, 4, -1, True, False, (1, 4), [0, 2], (True, 2), [1, False]])
    def test_invalid_degree_raises(self, sample_tensors, bad_degree):
        inv3 = compute_invariants(sample_tensors, max_degree=3, symmetry='SO3')
        with pytest.raises(ValueError, match="degree values must be 1, 2, or 3"):
            filter_by_degree(inv3, bad_degree)
