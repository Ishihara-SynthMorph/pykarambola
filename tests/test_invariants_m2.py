"""
Tests for Milestone 2: Degree-1 Invariants (Scalars) & Dependency Verification.

Tests verify:
- 8 independent scalars are produced
- Known identities: Tr(w102)/3 = w100 and Tr(w202)/3 = w200
- Deterministic label ordering
- Linear independence of the 8 scalars across diverse meshes
"""

import numpy as np
import pytest

from pykarambola.api import minkowski_tensors
from pykarambola.invariants import (
    SCALARS, RANK2_TENSORS, _RANK2_INDEPENDENT_TRACES, _DEGREE1_LABELS,
    trace_rank2, decompose_all, _degree1_scalars,
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


def _subdivide_icosphere(verts, faces):
    """Subdivide an icosphere."""
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


def _ellipsoid_mesh(a, b, c, subdivisions=2):
    """Build an ellipsoid mesh by scaling an icosphere."""
    verts, faces = _icosphere_mesh(radius=1.0, subdivisions=subdivisions)
    verts = verts * np.array([a, b, c])
    return verts, faces


def _random_convex_hull(n_points=50, seed=42):
    """Generate a random convex hull mesh."""
    from scipy.spatial import ConvexHull
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(n_points, 3))
    hull = ConvexHull(points)
    return points, hull.simplices.astype(np.int64)


# -----------------------------------------------------------------------------
# Test: Scalar dependency identities
# -----------------------------------------------------------------------------

class TestScalarDependencies:
    """Verify the known linear dependencies Tr(w102)/3 = w100 and Tr(w202)/3 = w200."""

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
        """Tr(w102)/3 = w100 (surface area identity)."""
        verts, faces = mesh_generator(*mesh_args)
        tensors = minkowski_tensors(verts, faces, compute='standard')

        w100 = tensors['w100']
        tr_w102 = trace_rank2(tensors['w102'])

        # Tr(w102) = ∫_S n·n dA = ∫_S dA = w100
        # trace_rank2 returns Tr(M)/3, so tr_w102 * 3 = w100
        assert tr_w102 * 3.0 == pytest.approx(w100, rel=1e-10), \
            f"Tr(w102) = {tr_w102 * 3.0} != w100 = {w100}"

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
        """Tr(w202)/3 = w200 (mean curvature identity)."""
        verts, faces = mesh_generator(*mesh_args)
        tensors = minkowski_tensors(verts, faces, compute='standard')

        w200 = tensors['w200']
        tr_w202 = trace_rank2(tensors['w202'])

        # Tr(w202) = ∫_S H (n·n) dA = ∫_S H dA = w200
        # trace_rank2 returns Tr(M)/3, so tr_w202 * 3 = w200
        assert tr_w202 * 3.0 == pytest.approx(w200, rel=1e-10), \
            f"Tr(w202) = {tr_w202 * 3.0} != w200 = {w200}"


# -----------------------------------------------------------------------------
# Test: _degree1_scalars function
# -----------------------------------------------------------------------------

class TestDegree1Scalars:
    """Test the _degree1_scalars function."""

    @pytest.fixture
    def box_decomposed(self):
        """Decomposed tensors from a box mesh."""
        verts, faces = _box_mesh(2.0, 3.0, 4.0)
        tensors = minkowski_tensors(verts, faces, compute='standard')
        return decompose_all(tensors), tensors

    def test_returns_8_scalars(self, box_decomposed):
        """Should return exactly 8 scalars."""
        decomposed, _ = box_decomposed
        scalars, labels = _degree1_scalars(decomposed)
        assert len(scalars) == 8
        assert scalars.shape == (8,)

    def test_returns_8_labels(self, box_decomposed):
        """Should return exactly 8 labels."""
        decomposed, _ = box_decomposed
        _, labels = _degree1_scalars(decomposed)
        assert len(labels) == 8

    def test_labels_match_expected(self, box_decomposed):
        """Labels should match the expected list."""
        decomposed, _ = box_decomposed
        _, labels = _degree1_scalars(decomposed)
        assert labels == list(_DEGREE1_LABELS)

    def test_first_four_are_base_scalars(self, box_decomposed):
        """s0-s3 should be w000, w100, w200, w300."""
        decomposed, tensors = box_decomposed
        scalars, _ = _degree1_scalars(decomposed)

        for i, name in enumerate(SCALARS):
            assert scalars[i] == pytest.approx(tensors[name], rel=1e-12), \
                f"s{i} = {scalars[i]} != {name} = {tensors[name]}"

    def test_s4_s7_are_independent_traces(self, box_decomposed):
        """s4-s7 should be Tr(w020)/3, Tr(w120)/3, Tr(w220)/3, Tr(w320)/3."""
        decomposed, tensors = box_decomposed
        scalars, _ = _degree1_scalars(decomposed)

        for i, name in enumerate(_RANK2_INDEPENDENT_TRACES):
            expected = trace_rank2(tensors[name])
            assert scalars[4 + i] == pytest.approx(expected, rel=1e-12), \
                f"s{4+i} = {scalars[4+i]} != Tr({name})/3 = {expected}"

    def test_deterministic_ordering(self):
        """Two calls should produce identical results."""
        verts, faces = _box_mesh(2.0, 3.0, 4.0)
        tensors = minkowski_tensors(verts, faces, compute='standard')
        decomposed = decompose_all(tensors)

        scalars1, labels1 = _degree1_scalars(decomposed)
        scalars2, labels2 = _degree1_scalars(decomposed)

        np.testing.assert_array_equal(scalars1, scalars2)
        assert labels1 == labels2


# -----------------------------------------------------------------------------
# Test: Linear independence of the 8 scalars
# -----------------------------------------------------------------------------

class TestScalarLinearIndependence:
    """Verify that the 8 scalars are linearly independent across diverse meshes."""

    def test_rank_across_diverse_meshes(self):
        """Feature matrix from diverse meshes should have rank 8."""
        meshes = [
            _box_mesh(2.0, 3.0, 4.0),
            _box_mesh(1.0, 1.0, 1.0),
            _box_mesh(1.0, 2.0, 3.0),
            _icosphere_mesh(1.0, 2),
            _icosphere_mesh(2.0, 2),
            _icosphere_mesh(0.5, 3),
            _ellipsoid_mesh(1.0, 2.0, 3.0, 2),
            _ellipsoid_mesh(2.0, 1.0, 1.0, 2),
            _random_convex_hull(50, 42),
            _random_convex_hull(80, 123),
        ]

        # Build feature matrix: N meshes x 8 scalars
        feature_matrix = []
        for verts, faces in meshes:
            tensors = minkowski_tensors(verts, faces, compute='standard')
            decomposed = decompose_all(tensors)
            scalars, _ = _degree1_scalars(decomposed)
            feature_matrix.append(scalars)

        X = np.array(feature_matrix)
        assert X.shape == (len(meshes), 8)

        # Compute rank (should be 8 if all scalars are independent)
        rank = np.linalg.matrix_rank(X, tol=1e-10)
        assert rank == 8, f"Expected rank 8, got {rank}. Scalars are not independent."

    def test_no_pairwise_linear_dependencies(self):
        """No two scalars should be linearly dependent (constant ratio)."""
        meshes = [
            _box_mesh(2.0, 3.0, 4.0),
            _box_mesh(1.0, 1.0, 1.0),
            _icosphere_mesh(1.0, 2),
            _ellipsoid_mesh(1.0, 2.0, 3.0, 2),
            _random_convex_hull(50, 42),
        ]

        feature_matrix = []
        for verts, faces in meshes:
            tensors = minkowski_tensors(verts, faces, compute='standard')
            decomposed = decompose_all(tensors)
            scalars, _ = _degree1_scalars(decomposed)
            feature_matrix.append(scalars)

        X = np.array(feature_matrix)

        # Check that no two columns are proportional
        for i in range(8):
            for j in range(i + 1, 8):
                col_i = X[:, i]
                col_j = X[:, j]

                # Skip if either column is all zeros
                if np.allclose(col_i, 0) or np.allclose(col_j, 0):
                    continue

                # Compute ratios where both values are nonzero
                mask = (np.abs(col_i) > 1e-12) & (np.abs(col_j) > 1e-12)
                if not np.any(mask):
                    continue

                ratios = col_i[mask] / col_j[mask]

                # If ratios are not all equal, columns are not proportional
                if not np.allclose(ratios, ratios[0], rtol=1e-6):
                    continue  # Not proportional, good

                # If we reach here, columns i and j are proportional (bad)
                assert False, f"Scalars {i} and {j} appear to be linearly dependent"


# -----------------------------------------------------------------------------
# Test: Excluded scalars would create dependencies
# -----------------------------------------------------------------------------

class TestExcludedScalarsCreateDependencies:
    """Verify that including Tr(w102) and Tr(w202) would create rank deficiency."""

    def test_including_tr_w102_w202_reduces_rank(self):
        """If we include Tr(w102)/3 and Tr(w202)/3, rank should be < 10."""
        meshes = [
            _box_mesh(2.0, 3.0, 4.0),
            _box_mesh(1.0, 1.0, 1.0),
            _box_mesh(1.0, 2.0, 3.0),
            _icosphere_mesh(1.0, 2),
            _icosphere_mesh(2.0, 2),
            _ellipsoid_mesh(1.0, 2.0, 3.0, 2),
            _random_convex_hull(50, 42),
            _random_convex_hull(80, 123),
        ]

        # Build 10-column feature matrix including Tr(w102) and Tr(w202)
        feature_matrix = []
        for verts, faces in meshes:
            tensors = minkowski_tensors(verts, faces, compute='standard')
            decomposed = decompose_all(tensors)

            # 8 independent scalars + 2 redundant ones
            scalars, _ = _degree1_scalars(decomposed)
            tr_w102 = decomposed[('w102', '0e')]  # = w100 / 3
            tr_w202 = decomposed[('w202', '0e')]  # = w200 / 3

            extended = np.concatenate([scalars, [tr_w102, tr_w202]])
            feature_matrix.append(extended)

        X = np.array(feature_matrix)
        assert X.shape == (len(meshes), 10)

        # Rank should be 8, not 10 (because of two exact dependencies)
        rank = np.linalg.matrix_rank(X, tol=1e-10)
        assert rank == 8, f"Expected rank 8 (two dependencies), got {rank}"
