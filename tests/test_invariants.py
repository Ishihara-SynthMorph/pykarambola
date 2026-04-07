"""
Tests for pykarambola.invariants: SO(3)-invariant scalar construction.

Organized by topic:
- TestTensorRegistry          — TENSOR_REGISTRY structure
- TestHarmonicDecomposition   — trace_rank2, traceless_rank2, decompose_all
- TestDegree1Invariants       — _degree1_scalars, scalar identities, linear independence
- TestDegree2Invariants       — _vector_dot_products, _frobenius_inner_products
- TestDegree3O3Invariants     — _quadratic_forms, _triple_traces
- TestDegree3SO3Pseudoscalars — _triple_vector_determinants, _commutator_pseudoscalars
- TestRotationalInvariance    — SO(3) invariance via public API (100 random rotations)
- TestReflectionBehavior      — O(3) invariance + SO(3) parity flip under reflections
- TestPublicAPI               — compute_invariants, compute_invariant_labels, error handling
- TestEndToEnd                — Clebsch-Gordan consistency, benchmarks, reproducibility
"""

import json
import time
import numpy as np
import pytest
from itertools import combinations_with_replacement
from scipy.spatial.transform import Rotation

from pykarambola.api import minkowski_tensors
from pykarambola.invariants import (
    TENSOR_REGISTRY, SCALARS, VECTORS, RANK2_TENSORS,
    _RANK2_INDEPENDENT_TRACES, _DEGREE1_LABELS,
    _VECTOR_ALIASES, _TRACELESS_ALIASES,
    trace_rank2, traceless_rank2, decompose_all,
    _degree1_scalars, _vector_dot_products, _frobenius_inner_products,
    _degree2_contractions, _quadratic_forms, _triple_traces,
    _degree3_o3_contractions, _triple_vector_determinants,
    _commutator_pseudoscalars, _degree3_so3_only_pseudoscalars,
    compute_invariants, compute_invariant_labels,
    _enumerate_invariant_contractions,
)


# =============================================================================
# Shared mesh generators
# =============================================================================

def _box_mesh(a, b, c):
    """Build a triangulated axis-aligned box centered at the origin."""
    ha, hb, hc = a / 2.0, b / 2.0, c / 2.0
    verts = np.array([
        [-ha, -hb, -hc], [ha, -hb, -hc], [ha, hb, -hc], [-ha, hb, -hc],
        [-ha, -hb, hc], [ha, -hb, hc], [ha, hb, hc], [-ha, hb, hc],
    ], dtype=np.float64)
    faces = np.array([
        [0, 3, 2], [0, 2, 1], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
        [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
    ], dtype=np.int64)
    return verts, faces


def _subdivide_icosphere(verts, faces):
    """Subdivide an icosphere by splitting each triangle into 4."""
    edge_midpoints = {}
    new_verts = list(verts)

    def get_midpoint(i, j):
        key = (min(i, j), max(i, j))
        if key in edge_midpoints:
            return edge_midpoints[key]
        mid = (verts[i] + verts[j]) / 2.0
        mid = mid / np.linalg.norm(mid)
        idx = len(new_verts)
        new_verts.append(mid)
        edge_midpoints[key] = idx
        return idx

    new_faces = []
    for v0, v1, v2 in faces:
        m01 = get_midpoint(v0, v1)
        m12 = get_midpoint(v1, v2)
        m20 = get_midpoint(v2, v0)
        new_faces.extend([
            [v0, m01, m20], [v1, m12, m01], [v2, m20, m12], [m01, m12, m20],
        ])
    return np.array(new_verts), np.array(new_faces, dtype=np.int64)


def _icosphere_mesh(radius=1.0, subdivisions=2):
    """Build an icosphere mesh."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=np.float64)
    verts = verts / np.linalg.norm(verts[0])
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int64)
    for _ in range(subdivisions):
        verts, faces = _subdivide_icosphere(verts, faces)
    return verts * radius, faces


def _ellipsoid_mesh(a, b, c, subdivisions=2):
    """Build an ellipsoid mesh by scaling an icosphere."""
    verts, faces = _icosphere_mesh(radius=1.0, subdivisions=subdivisions)
    return verts * np.array([a, b, c]), faces


def _shifted_ellipsoid_mesh(a, b, c, shift, subdivisions=2):
    """Build an ellipsoid mesh shifted from the origin."""
    verts, faces = _ellipsoid_mesh(a, b, c, subdivisions)
    return verts + np.array(shift), faces


def _torus_mesh(R=3.0, r=0.5, n_major=32, n_minor=16):
    """Build a torus mesh (non-convex)."""
    verts, faces = [], []
    for i in range(n_major):
        theta = 2 * np.pi * i / n_major
        for j in range(n_minor):
            phi = 2 * np.pi * j / n_minor
            verts.append([
                (R + r * np.cos(phi)) * np.cos(theta),
                (R + r * np.cos(phi)) * np.sin(theta),
                r * np.sin(phi),
            ])
    for i in range(n_major):
        for j in range(n_minor):
            i1, j1 = (i + 1) % n_major, (j + 1) % n_minor
            v0, v1 = i * n_minor + j, i1 * n_minor + j
            v2, v3 = i1 * n_minor + j1, i * n_minor + j1
            faces += [[v0, v1, v2], [v0, v2, v3]]
    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int64)


def _random_convex_hull(n_points=50, seed=42):
    """Generate a random convex hull mesh."""
    from scipy.spatial import ConvexHull
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(n_points, 3))
    hull = ConvexHull(points)
    return points, hull.simplices.astype(np.int64)


# =============================================================================
# Shared tensor transform helper
# =============================================================================

def _transform_tensors(tensors, R):
    """Apply transformation matrix R to all tensors (rotation or reflection)."""
    transformed = {}
    for name in SCALARS:
        transformed[name] = tensors[name]
    for name in VECTORS:
        transformed[name] = R @ tensors[name]
    for name in RANK2_TENSORS:
        M = np.asarray(tensors[name])
        transformed[name] = np.einsum('ia,jb,ab->ij', R, R, M)
    return transformed


# =============================================================================
# TestTensorRegistry
# =============================================================================

class TestTensorRegistry:
    """TENSOR_REGISTRY has correct structure and irrep labels."""

    def test_registry_has_14_entries(self):
        assert len(TENSOR_REGISTRY) == 14

    def test_scalars_count(self):
        assert len(SCALARS) == 4

    def test_vectors_count(self):
        assert len(VECTORS) == 4

    def test_rank2_count(self):
        assert len(RANK2_TENSORS) == 6

    def test_all_scalars_rank0(self):
        for name in SCALARS:
            assert TENSOR_REGISTRY[name]['rank'] == 0

    def test_all_vectors_rank1(self):
        for name in VECTORS:
            assert TENSOR_REGISTRY[name]['rank'] == 1

    def test_all_rank2_are_rank2(self):
        for name in RANK2_TENSORS:
            assert TENSOR_REGISTRY[name]['rank'] == 2

    def test_scalar_irreps(self):
        for name in SCALARS:
            assert TENSOR_REGISTRY[name]['irreps'] == ['0e']

    def test_vector_irreps(self):
        for name in VECTORS:
            assert TENSOR_REGISTRY[name]['irreps'] == ['1o']

    def test_rank2_irreps(self):
        for name in RANK2_TENSORS:
            assert TENSOR_REGISTRY[name]['irreps'] == ['0e', '2e']


# =============================================================================
# TestHarmonicDecomposition
# =============================================================================

class TestTraceAndTraceless:
    """trace_rank2 and traceless_rank2 basic correctness."""

    def test_trace_identity(self):
        assert trace_rank2(np.eye(3)) == pytest.approx(1.0)

    def test_trace_diagonal(self):
        assert trace_rank2(np.diag([2.0, 3.0, 4.0])) == pytest.approx(3.0)

    def test_traceless_identity(self):
        np.testing.assert_allclose(traceless_rank2(np.eye(3)), np.zeros((3, 3)), atol=1e-14)

    def test_traceless_is_traceless(self):
        M = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]], dtype=np.float64)
        assert np.trace(traceless_rank2(M)) == pytest.approx(0.0, abs=1e-14)

    def test_traceless_preserves_symmetry(self):
        M = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]], dtype=np.float64)
        T = traceless_rank2(M)
        np.testing.assert_allclose(T, T.T, atol=1e-14)


class TestDecompositionProperties:
    """Orthogonality, completeness, and norm preservation on various meshes."""

    @pytest.fixture(params=['box', 'icosphere', 'random_hull'])
    def mesh_tensors(self, request):
        if request.param == 'box':
            verts, faces = _box_mesh(2.0, 3.0, 4.0)
        elif request.param == 'icosphere':
            verts, faces = _icosphere_mesh(radius=1.0, subdivisions=2)
        else:
            verts, faces = _random_convex_hull(n_points=50, seed=42)
        return minkowski_tensors(verts, faces, compute='standard')

    def test_orthogonality(self, mesh_tensors):
        """Tr(T_tl) = 0 for all rank-2 tensors."""
        for name in RANK2_TENSORS:
            T_tl = traceless_rank2(np.asarray(mesh_tensors[name]))
            assert np.sum(T_tl * np.eye(3)) == pytest.approx(0.0, abs=1e-12), \
                f"Orthogonality failed for {name}"

    def test_completeness(self, mesh_tensors):
        """T_tl + t*I = M for all rank-2 tensors."""
        for name in RANK2_TENSORS:
            M = np.asarray(mesh_tensors[name])
            reconstructed = traceless_rank2(M) + trace_rank2(M) * np.eye(3)
            np.testing.assert_allclose(reconstructed, M, rtol=1e-12,
                err_msg=f"Completeness failed for {name}")

    def test_norm_preservation(self, mesh_tensors):
        """||M||_F^2 = ||T_tl||_F^2 + 3*t^2 for all rank-2 tensors."""
        for name in RANK2_TENSORS:
            M = np.asarray(mesh_tensors[name])
            t = trace_rank2(M)
            T_tl = traceless_rank2(M)
            np.testing.assert_allclose(
                np.sum(M**2), np.sum(T_tl**2) + 3.0 * t**2, rtol=1e-12,
                err_msg=f"Norm preservation failed for {name}")


class TestDecomposeAll:
    """decompose_all returns correctly structured output."""

    @pytest.fixture
    def decomposed(self):
        verts, faces = _box_mesh(2.0, 3.0, 4.0)
        tensors = minkowski_tensors(verts, faces, compute='standard')
        return decompose_all(tensors), tensors

    def test_scalar_keys(self, decomposed):
        d, _ = decomposed
        for name in SCALARS:
            assert (name, '0e') in d

    def test_vector_keys(self, decomposed):
        d, _ = decomposed
        for name in VECTORS:
            assert (name, '1o') in d

    def test_rank2_keys(self, decomposed):
        d, _ = decomposed
        for name in RANK2_TENSORS:
            assert (name, '0e') in d
            assert (name, '2e') in d

    def test_scalar_values_match(self, decomposed):
        d, tensors = decomposed
        for name in SCALARS:
            assert d[(name, '0e')] == pytest.approx(tensors[name])

    def test_vector_shapes(self, decomposed):
        d, _ = decomposed
        for name in VECTORS:
            assert d[(name, '1o')].shape == (3,)

    def test_rank2_trace_is_float(self, decomposed):
        d, _ = decomposed
        for name in RANK2_TENSORS:
            assert isinstance(d[(name, '0e')], float)

    def test_rank2_traceless_shape(self, decomposed):
        d, _ = decomposed
        for name in RANK2_TENSORS:
            assert d[(name, '2e')].shape == (3, 3)

    def test_total_key_count(self, decomposed):
        """4 scalars + 4 vectors + 6×2 rank-2 = 20 keys."""
        d, _ = decomposed
        assert len(d) == 4 + 4 + 6 + 6

    def test_missing_tensor_raises(self):
        with pytest.raises(KeyError):
            decompose_all({'w000': 1.0})

    @pytest.mark.parametrize("mesh_type,mesh_args", [
        ('box', (2.0, 3.0, 4.0)),
        ('box', (1.0, 1.0, 1.0)),
        ('icosphere', (1.0, 2)),
        ('icosphere', (2.5, 3)),
        ('random_hull', (50, 42)),
        ('random_hull', (100, 123)),
    ])
    def test_decomposition_tolerance(self, mesh_type, mesh_args):
        """All three decomposition properties hold to < 1e-12 relative error."""
        if mesh_type == 'box':
            verts, faces = _box_mesh(*mesh_args)
        elif mesh_type == 'icosphere':
            verts, faces = _icosphere_mesh(radius=mesh_args[0], subdivisions=mesh_args[1])
        else:
            verts, faces = _random_convex_hull(n_points=mesh_args[0], seed=mesh_args[1])
        tensors = minkowski_tensors(verts, faces, compute='standard')
        for name in RANK2_TENSORS:
            M = np.asarray(tensors[name])
            t = trace_rank2(M)
            T_tl = traceless_rank2(M)
            assert np.trace(T_tl) == pytest.approx(0.0, abs=1e-12)
            rel_err = np.linalg.norm(T_tl + t * np.eye(3) - M) / (np.linalg.norm(M) + 1e-14)
            assert rel_err < 1e-12, f"{mesh_type}: completeness failed for {name}"
            norm_M_sq = np.sum(M**2)
            expected = np.sum(T_tl**2) + 3.0 * t**2
            assert abs(norm_M_sq - expected) / (norm_M_sq + 1e-14) < 1e-12


# =============================================================================
# TestDegree1Invariants
# =============================================================================

class TestScalarDependencies:
    """Tr(w102)/3 = w100 and Tr(w202)/3 = w200 (known geometric identities)."""

    @pytest.mark.parametrize("mesh_generator,mesh_args", [
        (_box_mesh, (2.0, 3.0, 4.0)),
        (_box_mesh, (1.0, 1.0, 1.0)),
        (_icosphere_mesh, (1.0, 2)),
        (_icosphere_mesh, (2.5, 3)),
        (_ellipsoid_mesh, (1.0, 2.0, 3.0, 2)),
        (_random_convex_hull, (50, 42)),
        (_random_convex_hull, (100, 123)),
    ])
    def test_tr_w102_equals_w100(self, mesh_generator, mesh_args):
        """Tr(w102) = ∫ n·n dA = ∫ dA = w100."""
        verts, faces = mesh_generator(*mesh_args)
        tensors = minkowski_tensors(verts, faces, compute='standard')
        assert trace_rank2(tensors['w102']) * 3.0 == pytest.approx(tensors['w100'], rel=1e-10)

    @pytest.mark.parametrize("mesh_generator,mesh_args", [
        (_box_mesh, (2.0, 3.0, 4.0)),
        (_box_mesh, (1.0, 1.0, 1.0)),
        (_icosphere_mesh, (1.0, 2)),
        (_icosphere_mesh, (2.5, 3)),
        (_ellipsoid_mesh, (1.0, 2.0, 3.0, 2)),
        (_random_convex_hull, (50, 42)),
        (_random_convex_hull, (100, 123)),
    ])
    def test_tr_w202_equals_w200(self, mesh_generator, mesh_args):
        """Tr(w202) = ∫ H (n·n) dA = ∫ H dA = w200."""
        verts, faces = mesh_generator(*mesh_args)
        tensors = minkowski_tensors(verts, faces, compute='standard')
        assert trace_rank2(tensors['w202']) * 3.0 == pytest.approx(tensors['w200'], rel=1e-10)


class TestDegree1Scalars:
    """_degree1_scalars returns 8 correct, deterministic scalars."""

    @pytest.fixture
    def box_data(self):
        verts, faces = _box_mesh(2.0, 3.0, 4.0)
        tensors = minkowski_tensors(verts, faces, compute='standard')
        return decompose_all(tensors), tensors

    def test_returns_8_scalars(self, box_data):
        d, _ = box_data
        scalars, labels = _degree1_scalars(d)
        assert scalars.shape == (8,)
        assert len(labels) == 8

    def test_labels_match_expected(self, box_data):
        d, _ = box_data
        _, labels = _degree1_scalars(d)
        assert labels == list(_DEGREE1_LABELS)

    def test_first_four_are_base_scalars(self, box_data):
        d, tensors = box_data
        scalars, _ = _degree1_scalars(d)
        for i, name in enumerate(SCALARS):
            assert scalars[i] == pytest.approx(tensors[name], rel=1e-12)

    def test_s4_s7_are_independent_traces(self, box_data):
        d, tensors = box_data
        scalars, _ = _degree1_scalars(d)
        for i, name in enumerate(_RANK2_INDEPENDENT_TRACES):
            assert scalars[4 + i] == pytest.approx(trace_rank2(tensors[name]), rel=1e-12)

    def test_deterministic(self, box_data):
        d, _ = box_data
        s1, l1 = _degree1_scalars(d)
        s2, l2 = _degree1_scalars(d)
        np.testing.assert_array_equal(s1, s2)
        assert l1 == l2


class TestScalarLinearIndependence:
    """The 8 degree-1 scalars are linearly independent across diverse meshes."""

    def test_rank_across_diverse_meshes(self):
        meshes = [
            _box_mesh(2.0, 3.0, 4.0), _box_mesh(1.0, 1.0, 1.0), _box_mesh(1.0, 2.0, 3.0),
            _icosphere_mesh(1.0, 2), _icosphere_mesh(2.0, 2), _icosphere_mesh(0.5, 3),
            _ellipsoid_mesh(1.0, 2.0, 3.0, 2), _ellipsoid_mesh(2.0, 1.0, 1.0, 2),
            _random_convex_hull(50, 42), _random_convex_hull(80, 123),
        ]
        X = np.array([
            _degree1_scalars(decompose_all(minkowski_tensors(v, f, compute='standard')))[0]
            for v, f in meshes
        ])
        assert X.shape == (len(meshes), 8)
        assert np.linalg.matrix_rank(X, tol=1e-10) == 8

    def test_no_pairwise_linear_dependencies(self):
        meshes = [
            _box_mesh(2.0, 3.0, 4.0), _box_mesh(1.0, 1.0, 1.0),
            _icosphere_mesh(1.0, 2), _ellipsoid_mesh(1.0, 2.0, 3.0, 2),
            _random_convex_hull(50, 42),
        ]
        X = np.array([
            _degree1_scalars(decompose_all(minkowski_tensors(v, f, compute='standard')))[0]
            for v, f in meshes
        ])
        for i in range(8):
            for j in range(i + 1, 8):
                ci, cj = X[:, i], X[:, j]
                if np.allclose(ci, 0) or np.allclose(cj, 0):
                    continue
                mask = (np.abs(ci) > 1e-12) & (np.abs(cj) > 1e-12)
                if not np.any(mask):
                    continue
                ratios = ci[mask] / cj[mask]
                assert not np.allclose(ratios, ratios[0], rtol=1e-6), \
                    f"Scalars {i} and {j} appear linearly dependent"

    def test_including_tr_w102_w202_reduces_rank(self):
        """Adding Tr(w102)/3 and Tr(w202)/3 keeps rank at 8 (not 10)."""
        meshes = [
            _box_mesh(2.0, 3.0, 4.0), _box_mesh(1.0, 1.0, 1.0), _box_mesh(1.0, 2.0, 3.0),
            _icosphere_mesh(1.0, 2), _icosphere_mesh(2.0, 2),
            _ellipsoid_mesh(1.0, 2.0, 3.0, 2), _random_convex_hull(50, 42),
            _random_convex_hull(80, 123),
        ]
        rows = []
        for v, f in meshes:
            d = decompose_all(minkowski_tensors(v, f, compute='standard'))
            s, _ = _degree1_scalars(d)
            rows.append(np.concatenate([s, [d[('w102', '0e')], d[('w202', '0e')]]]))
        X = np.array(rows)
        assert X.shape[1] == 10
        assert np.linalg.matrix_rank(X, tol=1e-10) == 8


# =============================================================================
# TestDegree2Invariants
# =============================================================================

class TestDegree2Invariants:
    """Counts and symmetry properties of degree-2 contractions."""

    @pytest.fixture
    def box_decomposed(self):
        verts, faces = _box_mesh(2.0, 3.0, 4.0)
        return decompose_all(minkowski_tensors(verts, faces, compute='standard'))

    @pytest.fixture
    def ellipsoid_decomposed(self):
        verts, faces = _ellipsoid_mesh(1.0, 2.0, 3.0)
        return decompose_all(minkowski_tensors(verts, faces, compute='standard'))

    def test_dot_products_count(self, box_decomposed):
        inv, labels = _vector_dot_products(box_decomposed)
        assert len(inv) == 10
        assert len(labels) == 10

    def test_frobenius_products_count(self, box_decomposed):
        inv, labels = _frobenius_inner_products(box_decomposed)
        assert len(inv) == 21
        assert len(labels) == 21

    def test_degree2_total_count(self, box_decomposed):
        inv, labels = _degree2_contractions(box_decomposed)
        assert len(inv) == 31
        assert len(labels) == 31

    def test_dot_product_symmetry(self, ellipsoid_decomposed):
        """dot(vi, vj) == dot(vj, vi)."""
        vectors = [ellipsoid_decomposed[(name, '1o')] for name in VECTORS]
        nv = len(VECTORS)
        for i in range(nv):
            for j in range(nv):
                assert np.dot(vectors[i], vectors[j]) == pytest.approx(
                    np.dot(vectors[j], vectors[i]), rel=1e-14)

    def test_frobenius_symmetry(self, ellipsoid_decomposed):
        """frob(Ti, Tj) == frob(Tj, Ti)."""
        tl = [ellipsoid_decomposed[(name, '2e')] for name in RANK2_TENSORS]
        nt = len(RANK2_TENSORS)
        for i in range(nt):
            for j in range(nt):
                fij = np.einsum('ab,ab->', tl[i], tl[j])
                fji = np.einsum('ab,ab->', tl[j], tl[i])
                assert fij == pytest.approx(fji, rel=1e-14)


# =============================================================================
# TestDegree3O3Invariants
# =============================================================================

class TestDegree3O3Invariants:
    """Counts and symmetry properties of degree-3 O(3) contractions."""

    @pytest.fixture
    def box_decomposed(self):
        verts, faces = _box_mesh(2.0, 3.0, 4.0)
        return decompose_all(minkowski_tensors(verts, faces, compute='standard'))

    @pytest.fixture
    def ellipsoid_decomposed(self):
        verts, faces = _ellipsoid_mesh(1.0, 2.0, 3.0)
        return decompose_all(minkowski_tensors(verts, faces, compute='standard'))

    def test_quadratic_forms_count(self, box_decomposed):
        inv, labels = _quadratic_forms(box_decomposed)
        assert len(inv) == 60
        assert len(labels) == 60

    def test_triple_traces_count(self, box_decomposed):
        inv, labels = _triple_traces(box_decomposed)
        assert len(inv) == 56
        assert len(labels) == 56

    def test_degree3_o3_total_count(self, box_decomposed):
        inv, labels = _degree3_o3_contractions(box_decomposed)
        assert len(inv) == 116
        assert len(labels) == 116

    def test_burnside_formula(self):
        """C(n+2,3) = n(n+1)(n+2)/6 for n=6 gives 56."""
        n = len(RANK2_TENSORS)
        assert n * (n + 1) * (n + 2) // 6 == 56

    def test_multiset_enumeration(self):
        """We enumerate exactly C(8,3) = 56 multisets."""
        multisets = list(combinations_with_replacement(range(len(RANK2_TENSORS)), 3))
        assert len(multisets) == 56

    def test_quadratic_form_symmetry(self, ellipsoid_decomposed):
        """v_i^T T_k v_j == v_j^T T_k v_i for symmetric T_k."""
        nv, nt = len(VECTORS), len(RANK2_TENSORS)
        vectors = [ellipsoid_decomposed[(_VECTOR_ALIASES[f'v{i}'], '1o')] for i in range(nv)]
        traceless = [ellipsoid_decomposed[(_TRACELESS_ALIASES[f'T{k}'], '2e')] for k in range(nt)]
        for k in range(nt):
            T_k = traceless[k]
            for i in range(nv):
                for j in range(nv):
                    qf_ij = np.einsum('a,ab,b->', vectors[i], T_k, vectors[j])
                    qf_ji = np.einsum('a,ab,b->', vectors[j], T_k, vectors[i])
                    assert qf_ij == pytest.approx(qf_ji, rel=1e-14)


# =============================================================================
# TestDegree3SO3Pseudoscalars
# =============================================================================

class TestDegree3SO3Pseudoscalars:
    """Counts and non-triviality of degree-3 SO(3)-only pseudo-scalars."""

    @pytest.fixture
    def box_decomposed(self):
        verts, faces = _box_mesh(2.0, 3.0, 4.0)
        return decompose_all(minkowski_tensors(verts, faces, compute='standard'))

    @pytest.fixture
    def shifted_decomposed(self):
        verts, faces = _shifted_ellipsoid_mesh(1.0, 2.0, 3.0, [0.5, 0.3, 0.2])
        return decompose_all(minkowski_tensors(verts, faces, compute='standard'))

    def test_triple_vector_det_count(self, box_decomposed):
        inv, labels = _triple_vector_determinants(box_decomposed)
        assert len(inv) == 4
        assert len(labels) == 4

    def test_commutator_pseudoscalars_count(self, box_decomposed):
        inv, labels = _commutator_pseudoscalars(box_decomposed)
        assert len(inv) == 60
        assert len(labels) == 60

    def test_so3_only_total_count(self, box_decomposed):
        inv, labels = _degree3_so3_only_pseudoscalars(box_decomposed)
        assert len(inv) == 64
        assert len(labels) == 64

    def test_triple_vector_det_behavior(self, shifted_decomposed):
        """Dets may be near-zero for convex shapes (collinear centroid vectors)
        but must have the correct count and labels."""
        inv, labels = _triple_vector_determinants(shifted_decomposed)
        assert len(inv) == 4
        assert len(labels) == 4

    def test_commutator_pseudoscalars_nonzero_on_shifted_ellipsoid(self, shifted_decomposed):
        """At least one commutator pseudo-scalar is nonzero on an asymmetric mesh."""
        inv, _ = _commutator_pseudoscalars(shifted_decomposed)
        assert not np.allclose(inv, 0, atol=1e-14)


# =============================================================================
# TestRotationalInvariance
# =============================================================================

class TestRotationalInvariance:
    """Full invariant vector is stable under 100 random SO(3) rotations."""

    @pytest.mark.parametrize("mesh_type,symmetry", [
        ('box', 'O3'), ('box', 'SO3'),
        ('icosphere', 'O3'), ('icosphere', 'SO3'),
        ('ellipsoid', 'O3'), ('ellipsoid', 'SO3'),
        ('torus', 'O3'), ('torus', 'SO3'),
    ])
    def test_100_random_rotations(self, mesh_type, symmetry):
        if mesh_type == 'box':
            verts, faces = _box_mesh(2.0, 3.0, 4.0)
        elif mesh_type == 'icosphere':
            verts, faces = _icosphere_mesh(1.0, 2)
        elif mesh_type == 'ellipsoid':
            verts, faces = _ellipsoid_mesh(1.0, 2.0, 3.0, 2)
        else:
            verts, faces = _torus_mesh(R=3.0, r=0.5)

        tensors_ref = minkowski_tensors(verts, faces, compute='standard')
        inv_ref = compute_invariants(tensors_ref, max_degree=3, symmetry=symmetry)

        max_deviation = 0.0
        for trial in range(100):
            R = Rotation.random(random_state=trial).as_matrix()
            inv_rot = compute_invariants(
                _transform_tensors(tensors_ref, R), max_degree=3, symmetry=symmetry)
            ref_norm = np.linalg.norm(inv_ref)
            dev = (np.linalg.norm(inv_rot - inv_ref) / ref_norm
                   if ref_norm > 1e-14 else np.linalg.norm(inv_rot - inv_ref))
            max_deviation = max(max_deviation, dev)

        assert max_deviation < 1e-8, \
            f"Max deviation {max_deviation:.2e} for {mesh_type}/{symmetry}"

    def test_per_invariant_max_deviation(self):
        """No single invariant deviates by more than 1e-8 across 100 rotations."""
        verts, faces = _ellipsoid_mesh(1.0, 2.0, 3.0)
        tensors_ref = minkowski_tensors(verts, faces, compute='standard')
        inv_ref = compute_invariants(tensors_ref, max_degree=3, symmetry='SO3')
        labels = compute_invariant_labels(max_degree=3, symmetry='SO3')

        per_inv_max = np.zeros(len(inv_ref))
        for trial in range(100):
            R = Rotation.random(random_state=trial).as_matrix()
            inv_rot = compute_invariants(
                _transform_tensors(tensors_ref, R), max_degree=3, symmetry='SO3')
            for i in range(len(inv_ref)):
                ref_val = abs(inv_ref[i])
                dev = (abs(inv_rot[i] - inv_ref[i]) / ref_val
                       if ref_val > 1e-14 else abs(inv_rot[i] - inv_ref[i]))
                per_inv_max[i] = max(per_inv_max[i], dev)

        worst = np.argmax(per_inv_max)
        assert per_inv_max[worst] < 1e-8, \
            f"Worst invariant '{labels[worst]}' has max deviation {per_inv_max[worst]:.2e}"


# =============================================================================
# TestReflectionBehavior
# =============================================================================

class TestReflectionBehavior:
    """O(3) invariants are unchanged; SO(3) pseudo-scalars flip sign under reflections."""

    @pytest.mark.parametrize("reflection", [
        np.diag([-1, 1, 1]),
        np.diag([1, -1, 1]),
        np.diag([1, 1, -1]),
        np.diag([-1, -1, 1]),
    ])
    def test_o3_invariants_unchanged(self, reflection):
        """O(3) invariants are identical after a reflection (det(R) = -1)."""
        verts, faces = _ellipsoid_mesh(1.0, 2.0, 3.0)
        tensors_ref = minkowski_tensors(verts, faces, compute='standard')
        inv_ref = compute_invariants(tensors_ref, max_degree=3, symmetry='O3')
        inv_refl = compute_invariants(
            _transform_tensors(tensors_ref, reflection), max_degree=3, symmetry='O3')
        np.testing.assert_allclose(inv_refl, inv_ref, rtol=1e-12)

    @pytest.mark.parametrize("reflection", [
        np.diag([-1, 1, 1]),
        np.diag([1, -1, 1]),
        np.diag([1, 1, -1]),
    ])
    def test_triple_vector_det_flips_sign(self, reflection):
        """Triple vector determinants flip sign under reflection."""
        verts, faces = _shifted_ellipsoid_mesh(1.0, 2.0, 3.0, [0.5, 0.3, 0.2])
        tensors_ref = minkowski_tensors(verts, faces, compute='standard')
        d_ref = decompose_all(tensors_ref)
        det_ref, _ = _triple_vector_determinants(d_ref)
        det_refl, _ = _triple_vector_determinants(
            decompose_all(_transform_tensors(tensors_ref, reflection)))
        np.testing.assert_allclose(det_refl, -det_ref, rtol=1e-12)

    @pytest.mark.parametrize("reflection", [
        np.diag([-1, 1, 1]),
        np.diag([1, -1, 1]),
        np.diag([1, 1, -1]),
    ])
    def test_commutator_pseudoscalars_flip_sign(self, reflection):
        """Commutator pseudo-scalars flip sign under reflection."""
        verts, faces = _shifted_ellipsoid_mesh(1.0, 2.0, 3.0, [0.5, 0.3, 0.2])
        tensors_ref = minkowski_tensors(verts, faces, compute='standard')
        d_ref = decompose_all(tensors_ref)
        comm_ref, _ = _commutator_pseudoscalars(d_ref)
        comm_refl, _ = _commutator_pseudoscalars(
            decompose_all(_transform_tensors(tensors_ref, reflection)))
        np.testing.assert_allclose(comm_refl, -comm_ref, rtol=1e-12)

    @pytest.mark.parametrize("reflection", [
        np.diag([-1, 1, 1]),
        np.diag([1, -1, 1]),
        np.diag([1, 1, -1]),
    ])
    def test_all_so3_pseudoscalars_flip_sign(self, reflection):
        """All SO(3)-only pseudo-scalars flip sign under reflection."""
        verts, faces = _shifted_ellipsoid_mesh(1.0, 2.0, 3.0, [0.5, 0.3, 0.2])
        tensors_ref = minkowski_tensors(verts, faces, compute='standard')
        d_ref = decompose_all(tensors_ref)
        ps_ref, _ = _degree3_so3_only_pseudoscalars(d_ref)
        ps_refl, _ = _degree3_so3_only_pseudoscalars(
            decompose_all(_transform_tensors(tensors_ref, reflection)))
        np.testing.assert_allclose(ps_refl, -ps_ref, rtol=1e-12)


# =============================================================================
# TestPublicAPI
# =============================================================================

class TestAPIContract:
    """len(compute_invariants()) == len(compute_invariant_labels()) for all parameters."""

    @pytest.fixture
    def tensors(self):
        verts, faces = _box_mesh(2.0, 3.0, 4.0)
        return minkowski_tensors(verts, faces, compute='standard')

    @pytest.mark.parametrize("max_degree,symmetry", [
        (1, 'O3'), (1, 'SO3'),
        (2, 'O3'), (2, 'SO3'),
        (3, 'O3'), (3, 'SO3'),
    ])
    def test_invariants_and_labels_same_length(self, tensors, max_degree, symmetry):
        inv = compute_invariants(tensors, max_degree=max_degree, symmetry=symmetry)
        labels = compute_invariant_labels(max_degree=max_degree, symmetry=symmetry)
        assert len(inv) == len(labels)


class TestMaxDegreeGating:
    """max_degree controls invariant count exactly."""

    @pytest.fixture
    def tensors(self):
        verts, faces = _box_mesh(2.0, 3.0, 4.0)
        return minkowski_tensors(verts, faces, compute='standard')

    def test_degree1_count(self, tensors):
        assert len(compute_invariants(tensors, max_degree=1)) == 8

    def test_degree2_count(self, tensors):
        assert len(compute_invariants(tensors, max_degree=2)) == 39

    def test_degree3_o3_count(self, tensors):
        assert len(compute_invariants(tensors, max_degree=3, symmetry='O3')) == 155

    def test_degree3_so3_count(self, tensors):
        assert len(compute_invariants(tensors, max_degree=3, symmetry='SO3')) == 219


class TestDeterministicOrdering:
    """Labels and invariants are deterministic across calls."""

    def test_labels_deterministic(self):
        labels1 = compute_invariant_labels(max_degree=3, symmetry='SO3')
        labels2 = compute_invariant_labels(max_degree=3, symmetry='SO3')
        assert labels1 == labels2

    def test_invariants_deterministic(self):
        verts, faces = _box_mesh(2.0, 3.0, 4.0)
        tensors = minkowski_tensors(verts, faces, compute='standard')
        inv1 = compute_invariants(tensors, max_degree=3, symmetry='SO3')
        inv2 = compute_invariants(tensors, max_degree=3, symmetry='SO3')
        np.testing.assert_array_equal(inv1, inv2)

    def test_all_labels_unique(self):
        labels = compute_invariant_labels(max_degree=3, symmetry='SO3')
        assert len(labels) == len(set(labels))


class TestTranslationBehavior:
    """Document translation (co)variance of invariants."""

    def test_translation_changes_invariants(self):
        verts, faces = _ellipsoid_mesh(1.0, 2.0, 3.0)
        tensors_orig = minkowski_tensors(verts, faces, compute='standard')
        tensors_shifted = minkowski_tensors(verts + [10.0, 20.0, 30.0], faces, compute='standard')
        inv_orig = compute_invariants(tensors_orig, symmetry='O3')
        inv_shifted = compute_invariants(tensors_shifted, symmetry='O3')
        assert not np.allclose(inv_orig, inv_shifted)

    def test_translation_invariant_subset(self):
        """s0-s3 (built from w000, w100, w200, w300) are translation-invariant."""
        verts, faces = _ellipsoid_mesh(1.0, 2.0, 3.0)
        tensors_orig = minkowski_tensors(verts, faces, compute='standard')
        tensors_shifted = minkowski_tensors(verts + [10.0, 20.0, 30.0], faces, compute='standard')
        inv_orig = compute_invariants(tensors_orig, max_degree=2, symmetry='O3')
        inv_shifted = compute_invariants(tensors_shifted, max_degree=2, symmetry='O3')
        np.testing.assert_allclose(inv_orig[:4], inv_shifted[:4], rtol=1e-10)

    def test_s4_s7_are_translation_covariant(self):
        """s4-s7 (traces of w020-w320) change under translation."""
        verts, faces = _ellipsoid_mesh(1.0, 2.0, 3.0)
        tensors_orig = minkowski_tensors(verts, faces, compute='standard')
        tensors_shifted = minkowski_tensors(verts + [10.0, 20.0, 30.0], faces, compute='standard')
        inv_orig = compute_invariants(tensors_orig, max_degree=1, symmetry='O3')
        inv_shifted = compute_invariants(tensors_shifted, max_degree=1, symmetry='O3')
        assert not np.allclose(inv_orig[4:8], inv_shifted[4:8])


class TestEdgeCases:
    """Error handling for invalid inputs."""

    @pytest.fixture
    def tensors(self):
        verts, faces = _box_mesh(2.0, 3.0, 4.0)
        return minkowski_tensors(verts, faces, compute='standard')

    def test_invalid_max_degree_raises(self, tensors):
        with pytest.raises(ValueError, match="max_degree must be 1, 2, or 3"):
            compute_invariants(tensors, max_degree=0)
        with pytest.raises(ValueError, match="max_degree must be 1, 2, or 3"):
            compute_invariants(tensors, max_degree=4)

    def test_invalid_symmetry_raises(self, tensors):
        with pytest.raises(ValueError, match="symmetry must be 'SO3' or 'O3'"):
            compute_invariants(tensors, symmetry='invalid')

    def test_missing_tensor_raises(self):
        with pytest.raises(KeyError):
            compute_invariants({'w000': 1.0})

    def test_labels_invalid_max_degree_raises(self):
        with pytest.raises(ValueError):
            compute_invariant_labels(max_degree=0)

    def test_labels_invalid_symmetry_raises(self):
        with pytest.raises(ValueError):
            compute_invariant_labels(symmetry='invalid')

    def test_labels_tensors_dict_validation(self):
        """tensors_dict=incomplete raises KeyError."""
        with pytest.raises(KeyError):
            compute_invariant_labels(tensors_dict={'w000': 1.0})


class TestEnumerationHelper:
    """_enumerate_invariant_contractions returns correct counts."""

    def test_o3_counts(self):
        counts = _enumerate_invariant_contractions(symmetry='O3')
        assert counts['degree1_scalars'] == 8
        assert counts['degree2_dot_products'] == 10
        assert counts['degree2_frobenius'] == 21
        assert counts['degree3_quadratic_forms'] == 60
        assert counts['degree3_triple_traces'] == 56
        assert counts['total'] == 155

    def test_so3_counts(self):
        counts = _enumerate_invariant_contractions(symmetry='SO3')
        assert counts['degree1_scalars'] == 8
        assert counts['degree2_dot_products'] == 10
        assert counts['degree2_frobenius'] == 21
        assert counts['degree3_quadratic_forms'] == 60
        assert counts['degree3_triple_traces'] == 56
        assert counts['degree3_triple_vector_dets'] == 4
        assert counts['degree3_commutator_pseudoscalars'] == 60
        assert counts['total'] == 219


# =============================================================================
# TestEndToEnd
# =============================================================================

class TestClebschGordanConsistency:
    """All contractions satisfy Wigner-3j selection rules."""

    def test_registry_irreps_valid(self):
        valid_irreps = {'0e', '1o', '2e'}
        for name, info in TENSOR_REGISTRY.items():
            for irrep in info['irreps']:
                assert irrep in valid_irreps, f"Invalid irrep '{irrep}' in {name}"

    def test_scalar_contractions_are_0e(self):
        verts, faces = _box_mesh(2.0, 3.0, 4.0)
        d = decompose_all(minkowski_tensors(verts, faces, compute='standard'))
        for name in SCALARS:
            assert isinstance(d[(name, '0e')], float)
        for name in _RANK2_INDEPENDENT_TRACES:
            assert isinstance(d[(name, '0e')], float)

    def test_output_dtype_is_float64(self):
        verts, faces = _box_mesh(2.0, 3.0, 4.0)
        tensors = minkowski_tensors(verts, faces, compute='standard')
        assert compute_invariants(tensors, max_degree=3).dtype == np.float64


class TestPerformanceBenchmark:
    """compute_invariants completes within 1 s per call on typical meshes."""

    @pytest.mark.parametrize("subdivisions,expected_faces", [
        (1, 80), (2, 320), (3, 1280),
    ])
    def test_benchmark_icosphere(self, subdivisions, expected_faces):
        verts, faces = _icosphere_mesh(radius=1.0, subdivisions=subdivisions)
        assert len(faces) == expected_faces
        tensors = minkowski_tensors(verts, faces, compute='standard')
        start = time.perf_counter()
        for _ in range(10):
            compute_invariants(tensors, max_degree=3, symmetry='SO3')
        elapsed = (time.perf_counter() - start) / 10
        assert elapsed < 1.0, f"compute_invariants took {elapsed:.3f}s on {len(faces)} triangles"


class TestInvariantReproducibility:
    """Invariants are bitwise identical after tensor serialization round-trip."""

    def test_serialize_deserialize(self):
        verts, faces = _ellipsoid_mesh(1.0, 2.0, 3.0)
        tensors = minkowski_tensors(verts, faces, compute='standard')
        inv1 = compute_invariants(tensors, max_degree=3, symmetry='SO3')

        serialized = {k: v.tolist() if isinstance(v, np.ndarray) else v
                      for k, v in tensors.items()}
        loaded = json.loads(json.dumps(serialized))
        deserialized = {k: np.array(v, dtype=np.float64) if isinstance(v, list) else v
                        for k, v in loaded.items()}

        inv2 = compute_invariants(deserialized, max_degree=3, symmetry='SO3')
        np.testing.assert_array_equal(inv1, inv2)


class TestSummaryTableVerification:
    """Invariant counts match the issue #102 specification table."""

    def test_o3_total_155(self):
        assert _enumerate_invariant_contractions(symmetry='O3')['total'] == 155

    def test_so3_total_219(self):
        assert _enumerate_invariant_contractions(symmetry='SO3')['total'] == 219

    def test_breakdown_matches_issue(self):
        counts = _enumerate_invariant_contractions(symmetry='SO3')
        assert counts['degree1_scalars'] == 8
        assert counts['degree2_dot_products'] == 10
        assert counts['degree2_frobenius'] == 21
        assert counts['degree3_quadratic_forms'] == 60
        assert counts['degree3_triple_traces'] == 56
        assert counts['degree3_triple_vector_dets'] == 4
        assert counts['degree3_commutator_pseudoscalars'] == 60
