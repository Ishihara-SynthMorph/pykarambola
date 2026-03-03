"""
Tests for non-convex mesh handling and degenerate-triangle detection.

Covers:
  #44 — zero-area triangle warning in Triangulation._precompute
  #50 — w202 sign convention for concave edges
"""

import math
import warnings

import numpy as np
import pytest

from pykarambola.triangulation import Triangulation, LABEL_UNASSIGNED
from pykarambola.minkowski import (
    calculate_w200, calculate_w300, calculate_w202,
)
from pykarambola.eigensystem import calculate_eigensystem


def _notched_box_mesh():
    """L-shaped prism: outer 4×3×4 box with a 2×3×2 corner notch removed.

    Cross-section in the x-z plane is an L-shape; extruded 3 units along y.
    Volume = 4*4*3 - 2*2*3 = 36.

    Exactly one concave vertical edge runs from vertex 1=(2,0,2) to
    vertex 7=(2,3,2), length=3, dihedral angle ≈ -π/2.  All other edges
    are convex with dihedral angle = +π/2.

    Analytical w200 = 11π/3.
    """
    verts = np.array([
        # bottom face (y=0), L-shape
        [0, 0, 2],  # 0  A'
        [2, 0, 2],  # 1  B'  ← bottom of concave edge
        [2, 0, 0],  # 2  C'
        [4, 0, 0],  # 3  D'
        [4, 0, 4],  # 4  E'
        [0, 0, 4],  # 5  F'
        # top face (y=3)
        [0, 3, 2],  # 6  A
        [2, 3, 2],  # 7  B   ← top of concave edge
        [2, 3, 0],  # 8  C
        [4, 3, 0],  # 9  D
        [4, 3, 4],  # 10 E
        [0, 3, 4],  # 11 F
    ], dtype=np.float64)

    faces = np.array([
        # bottom face (outward normal -y)
        [0, 1, 5], [1, 4, 5], [1, 2, 3], [1, 3, 4],
        # top face (outward normal +y)
        [6, 11, 7], [7, 11, 10], [7, 10, 9], [7, 9, 8],
        # inner wall 1: x=2 plane (outward normal -x) — shares concave edge 1-7
        [1, 7, 2], [2, 7, 8],
        # inner wall 2: z=2 plane (outward normal -z) — shares concave edge 1-7
        [0, 6, 1], [1, 6, 7],
        # front face: z=0 (outward normal -z)
        [2, 8, 3], [3, 8, 9],
        # right face: x=4 (outward normal +x)
        [3, 9, 4], [4, 9, 10],
        # back face: z=4 (outward normal +z)
        [4, 10, 5], [5, 10, 11],
        # left face: x=0 (outward normal -x)
        [0, 5, 6], [5, 11, 6],
    ], dtype=np.int64)

    return verts, faces


def _sorted_eigenvalues(w_matrix, label):
    eig = calculate_eigensystem(w_matrix)
    return sorted(eig[label].result.eigen_values, key=abs)


class TestNonConvex:
    """#50: w202 sign convention for concave edges on an L-shaped prism."""

    @pytest.fixture(autouse=True)
    def setup(self):
        verts, faces = _notched_box_mesh()
        self.surface = Triangulation.from_arrays(verts, faces)
        self.label = 0
        self.pi = math.pi

    def test_w300_euler_characteristic(self):
        """L-prism is genus-0 (topologically a sphere) → Euler characteristic = 4π/3."""
        w300 = calculate_w300(self.surface)
        assert w300[self.label].result == pytest.approx(4 * self.pi / 3, rel=1e-4)

    def test_w200_mean_curvature(self):
        """Analytical w200 for the L-prism = 11π/3.

        Derivation: each edge contributes alpha * L / 6.
        Five convex vertical edges (L=3, alpha=+π/2) + one concave (L=3, alpha=-π/2)
        give net π.  Six bottom + six top edges (all convex, total length 16 each)
        contribute 2 × (π/2 × 16 / 6) = 8π/3.  Total = π + 8π/3 = 11π/3.
        """
        w200 = calculate_w200(self.surface)
        assert w200[self.label].result == pytest.approx(11 * self.pi / 3, rel=1e-4)

    def test_concave_edge_has_negative_dihedral_angle(self):
        """The concave inner-corner edge must be stored with a negative dihedral angle.

        The edge from vertex 1=(2,0,2) to vertex 7=(2,3,2) is shared by inner wall 1
        and inner wall 2; their outward normals are (-1,0,0) and (0,0,-1) respectively,
        forming a concave 'valley'.  Both triangles record alpha ≈ -π/2.
        """
        alphas = self.surface._dihedral_angles.flatten()
        neg = alphas[alphas < 0]
        assert len(neg) == 2, "Expected exactly 2 negative dihedral angles (one concave edge)"
        np.testing.assert_allclose(neg, -self.pi / 2, atol=1e-6)

    def test_w202_no_nan(self):
        """w202 entries must all be finite on the non-convex mesh."""
        w202 = calculate_w202(self.surface)
        mat = w202[self.label].result
        for i in range(3):
            for j in range(i + 1):
                assert np.isfinite(mat[i, j]), f"w202[{i},{j}] is not finite"

    def test_w202_eigenvalues(self):
        """w202 eigenvalues must match C++ karambola reference values.

        TODO: fill in expected values after running C++ karambola on the
        notched-box mesh and replace the pytest.skip call below.

        Pykarambola computes (sorted): ≈ [3.6652, 3.6652, 4.1888]
        """
        pytest.skip("Reference values pending C++ karambola validation")
        # Once C++ values are available, replace with:
        # expected = sorted([TODO, TODO, TODO])
        # actual = _sorted_eigenvalues(calculate_w202(self.surface), self.label)
        # np.testing.assert_allclose(actual, expected, rtol=1e-4)


class TestDegenerateTriangleWarning:
    """#44: Triangulation._precompute warns when zero-area triangles are present."""

    def test_zero_area_triangle_warns(self):
        """A mesh containing collinear vertices (zero-area face) must emit a UserWarning
        mentioning 'degenerate'."""
        # Vertex 0,1,2 are collinear → face [0,1,2] has zero area
        verts = np.array([[0,0,0],[1,0,0],[2,0,0],[0,0,1]], dtype=np.float64)
        faces = np.array([[0,1,2],[0,1,3]], dtype=np.int64)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Triangulation.from_arrays(verts, faces)

        assert any(issubclass(w.category, UserWarning) for w in caught)
        assert any("degenerate" in str(w.message).lower() for w in caught)

    def test_valid_mesh_no_degenerate_warning(self):
        """A clean mesh with no zero-area triangles must not produce a degenerate warning."""
        verts, faces = _notched_box_mesh()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Triangulation.from_arrays(verts, faces)

        assert not any("degenerate" in str(w.message).lower() for w in caught)

    def test_warning_count_matches_degenerate_triangles(self):
        """Warning message must report the exact count of zero-area triangles."""
        # Two degenerate triangles: [0,1,2] and [0,1,3] are both collinear
        verts = np.array([[0,0,0],[1,0,0],[2,0,0],[3,0,0],[0,0,1]], dtype=np.float64)
        faces = np.array([[0,1,2],[0,1,3],[0,1,4]], dtype=np.int64)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Triangulation.from_arrays(verts, faces)

        degenerate_warnings = [w for w in caught
                               if issubclass(w.category, UserWarning)
                               and "degenerate" in str(w.message).lower()]
        assert len(degenerate_warnings) == 1
        assert "2" in str(degenerate_warnings[0].message)
