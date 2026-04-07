"""
Tests for Milestone 1: Tensor Registry & Harmonic Decomposition.

Tests verify:
- Orthogonality: traceless component has zero trace
- Completeness: trace + traceless recovers original
- Norm preservation: ||M||_F^2 = ||T_tl||_F^2 + 3 * trace^2
"""

import numpy as np
import pytest

from pykarambola.api import minkowski_tensors
from pykarambola.invariants import (
    TENSOR_REGISTRY, SCALARS, VECTORS, RANK2_TENSORS,
    trace_rank2, traceless_rank2, decompose_all,
)


# -----------------------------------------------------------------------------
# Test fixtures: mesh generation
# -----------------------------------------------------------------------------

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


def _icosphere_mesh(radius=1.0, subdivisions=2):
    """Build an icosphere mesh with given radius and subdivisions."""
    # Golden ratio
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    # Icosahedron vertices
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=np.float64)
    # Normalize to unit sphere
    verts = verts / np.linalg.norm(verts[0])
    # Icosahedron faces
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int64)
    # Subdivide
    for _ in range(subdivisions):
        verts, faces = _subdivide_icosphere(verts, faces)
    # Scale to desired radius
    verts = verts * radius
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
        mid = mid / np.linalg.norm(mid)  # project to unit sphere
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
            [v0, m01, m20],
            [v1, m12, m01],
            [v2, m20, m12],
            [m01, m12, m20],
        ])
    return np.array(new_verts), np.array(new_faces, dtype=np.int64)


def _random_convex_hull(n_points=50, seed=42):
    """Generate a random convex hull mesh."""
    from scipy.spatial import ConvexHull
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(n_points, 3))
    hull = ConvexHull(points)
    return points, hull.simplices.astype(np.int64)


# -----------------------------------------------------------------------------
# Test: TENSOR_REGISTRY structure
# -----------------------------------------------------------------------------

class TestTensorRegistry:
    """Test that TENSOR_REGISTRY has correct structure."""

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


# -----------------------------------------------------------------------------
# Test: trace_rank2 and traceless_rank2 functions
# -----------------------------------------------------------------------------

class TestTraceAndTraceless:
    """Test trace_rank2 and traceless_rank2 functions."""

    def test_trace_identity(self):
        """Trace of identity matrix is 1."""
        assert trace_rank2(np.eye(3)) == pytest.approx(1.0)

    def test_trace_diagonal(self):
        """Trace of diagonal matrix."""
        M = np.diag([2.0, 3.0, 4.0])
        assert trace_rank2(M) == pytest.approx(3.0)  # (2+3+4)/3

    def test_traceless_identity(self):
        """Traceless part of identity is zero."""
        result = traceless_rank2(np.eye(3))
        np.testing.assert_allclose(result, np.zeros((3, 3)), atol=1e-14)

    def test_traceless_is_traceless(self):
        """Traceless part has zero trace."""
        M = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]], dtype=np.float64)
        T_tl = traceless_rank2(M)
        assert np.trace(T_tl) == pytest.approx(0.0, abs=1e-14)

    def test_traceless_preserves_symmetry(self):
        """Traceless part of symmetric matrix is symmetric."""
        M = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]], dtype=np.float64)
        T_tl = traceless_rank2(M)
        np.testing.assert_allclose(T_tl, T_tl.T, atol=1e-14)


# -----------------------------------------------------------------------------
# Test: Harmonic decomposition properties
# -----------------------------------------------------------------------------

class TestHarmonicDecomposition:
    """Test the three decomposition properties on various meshes."""

    @pytest.fixture(params=['box', 'icosphere', 'random_hull'])
    def mesh_tensors(self, request):
        """Fixture providing Minkowski tensors for different mesh types."""
        if request.param == 'box':
            verts, faces = _box_mesh(2.0, 3.0, 4.0)
        elif request.param == 'icosphere':
            verts, faces = _icosphere_mesh(radius=1.0, subdivisions=2)
        else:
            verts, faces = _random_convex_hull(n_points=50, seed=42)
        return minkowski_tensors(verts, faces, compute='standard')

    def test_orthogonality(self, mesh_tensors):
        """Traceless component is orthogonal to identity: Tr(T_tl) = 0."""
        for name in RANK2_TENSORS:
            M = np.asarray(mesh_tensors[name])
            T_tl = traceless_rank2(M)
            # Orthogonality: T_tl · I = sum_i T_tl[i,i] = Tr(T_tl) = 0
            inner_product = np.sum(T_tl * np.eye(3))
            assert inner_product == pytest.approx(0.0, abs=1e-12), \
                f"Orthogonality failed for {name}: Tr(T_tl) = {inner_product}"

    def test_completeness(self, mesh_tensors):
        """Trace + traceless reconstructs original: T_tl + t*I = M."""
        for name in RANK2_TENSORS:
            M = np.asarray(mesh_tensors[name])
            t = trace_rank2(M)
            T_tl = traceless_rank2(M)
            # M = T_tl + t*I where t = Tr(M)/3
            reconstructed = T_tl + t * np.eye(3)
            np.testing.assert_allclose(reconstructed, M, rtol=1e-12,
                err_msg=f"Completeness failed for {name}")

    def test_norm_preservation(self, mesh_tensors):
        """Frobenius norm is preserved: ||M||^2 = ||T_tl||^2 + 3*trace^2."""
        for name in RANK2_TENSORS:
            M = np.asarray(mesh_tensors[name])
            t = trace_rank2(M)
            T_tl = traceless_rank2(M)
            norm_M_sq = np.sum(M**2)
            norm_tl_sq = np.sum(T_tl**2)
            trace_contrib = 3.0 * t**2
            np.testing.assert_allclose(norm_M_sq, norm_tl_sq + trace_contrib, rtol=1e-12,
                err_msg=f"Norm preservation failed for {name}")


# -----------------------------------------------------------------------------
# Test: decompose_all function
# -----------------------------------------------------------------------------

class TestDecomposeAll:
    """Test decompose_all function."""

    @pytest.fixture
    def tensors_and_decomposed(self):
        """Fixture providing tensors and their decomposition."""
        verts, faces = _box_mesh(2.0, 3.0, 4.0)
        tensors = minkowski_tensors(verts, faces, compute='standard')
        decomposed = decompose_all(tensors)
        return tensors, decomposed

    def test_scalar_keys(self, tensors_and_decomposed):
        """Scalars produce (name, '0e') keys."""
        _, decomposed = tensors_and_decomposed
        for name in SCALARS:
            assert (name, '0e') in decomposed

    def test_vector_keys(self, tensors_and_decomposed):
        """Vectors produce (name, '1o') keys."""
        _, decomposed = tensors_and_decomposed
        for name in VECTORS:
            assert (name, '1o') in decomposed

    def test_rank2_keys(self, tensors_and_decomposed):
        """Rank-2 tensors produce both (name, '0e') and (name, '2e') keys."""
        _, decomposed = tensors_and_decomposed
        for name in RANK2_TENSORS:
            assert (name, '0e') in decomposed
            assert (name, '2e') in decomposed

    def test_scalar_values_match(self, tensors_and_decomposed):
        """Scalar irreps match original values."""
        tensors, decomposed = tensors_and_decomposed
        for name in SCALARS:
            assert decomposed[(name, '0e')] == pytest.approx(tensors[name])

    def test_vector_shapes(self, tensors_and_decomposed):
        """Vector irreps have shape (3,)."""
        _, decomposed = tensors_and_decomposed
        for name in VECTORS:
            assert decomposed[(name, '1o')].shape == (3,)

    def test_rank2_trace_shapes(self, tensors_and_decomposed):
        """Rank-2 trace irreps are floats."""
        _, decomposed = tensors_and_decomposed
        for name in RANK2_TENSORS:
            assert isinstance(decomposed[(name, '0e')], float)

    def test_rank2_traceless_shapes(self, tensors_and_decomposed):
        """Rank-2 traceless irreps have shape (3, 3)."""
        _, decomposed = tensors_and_decomposed
        for name in RANK2_TENSORS:
            assert decomposed[(name, '2e')].shape == (3, 3)

    def test_total_key_count(self, tensors_and_decomposed):
        """Total number of keys: 4 scalars + 4 vectors + 6*2 rank-2 = 20."""
        _, decomposed = tensors_and_decomposed
        # 4 scalars (0e) + 4 vectors (1o) + 6 rank-2 traces (0e) + 6 rank-2 traceless (2e)
        assert len(decomposed) == 4 + 4 + 6 + 6

    def test_missing_tensor_raises(self):
        """Missing required tensor raises KeyError."""
        incomplete = {'w000': 1.0}  # Missing most tensors
        with pytest.raises(KeyError):
            decompose_all(incomplete)


# -----------------------------------------------------------------------------
# Test: Decomposition tolerance on multiple mesh types
# -----------------------------------------------------------------------------

class TestDecompositionTolerance:
    """Test decomposition properties with specific tolerance requirements."""

    @pytest.mark.parametrize("mesh_type,mesh_args", [
        ('box', (2.0, 3.0, 4.0)),
        ('box', (1.0, 1.0, 1.0)),  # Unit cube
        ('icosphere', (1.0, 2)),
        ('icosphere', (2.5, 3)),
        ('random_hull', (50, 42)),
        ('random_hull', (100, 123)),
    ])
    def test_all_properties_within_tolerance(self, mesh_type, mesh_args):
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

            # Orthogonality
            assert np.trace(T_tl) == pytest.approx(0.0, abs=1e-12), \
                f"{mesh_type} {mesh_args}: orthogonality failed for {name}"

            # Completeness: M = T_tl + t*I where t = Tr(M)/3
            reconstructed = T_tl + t * np.eye(3)
            rel_err = np.linalg.norm(reconstructed - M) / (np.linalg.norm(M) + 1e-14)
            assert rel_err < 1e-12, \
                f"{mesh_type} {mesh_args}: completeness failed for {name} (rel_err={rel_err})"

            # Norm preservation
            norm_M_sq = np.sum(M**2)
            norm_tl_sq = np.sum(T_tl**2)
            expected = norm_tl_sq + 3.0 * t**2
            rel_err = abs(norm_M_sq - expected) / (norm_M_sq + 1e-14)
            assert rel_err < 1e-12, \
                f"{mesh_type} {mesh_args}: norm preservation failed for {name} (rel_err={rel_err})"
