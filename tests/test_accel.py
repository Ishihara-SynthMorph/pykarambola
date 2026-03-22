"""
Tests for the Cython acceleration module and its Python fallback.

Both ``_build_vertex_polygon_lookup`` and ``_build_polygon_polygon_lookup``
in ``Triangulation`` have two code paths: a compiled Cython path (when
``_accel`` is available) and a pure-Python fallback.  These tests verify:

1. The Python fallback path produces correct results (always runs).
2. The Cython path produces correct results (skipped if not compiled).
3. Both paths produce identical results on the same mesh (skipped if not compiled).
"""

import unittest.mock

import numpy as np
import pytest

import pykarambola.triangulation as tri_module
from pykarambola.triangulation import Triangulation, NEIGHBOUR_UNASSIGNED

# A 2x3x4 box — 8 vertices, 12 triangles, fully closed (manifold).
# Every edge is shared by exactly two triangles, so no NEIGHBOUR_UNASSIGNED
# entries should appear in the neighbour table.
_VERTS = np.array([
    [ 1.0, -1.5,  2.0],
    [-1.0, -1.5,  2.0],
    [ 1.0,  1.5,  2.0],
    [-1.0,  1.5,  2.0],
    [-1.0, -1.5, -2.0],
    [ 1.0, -1.5, -2.0],
    [-1.0,  1.5, -2.0],
    [ 1.0,  1.5, -2.0],
], dtype=np.float64)

_FACES = np.array([
    [3, 1, 0], [2, 3, 0],
    [7, 5, 4], [6, 7, 4],
    [2, 7, 6], [3, 2, 6],
    [1, 4, 5], [0, 1, 5],
    [2, 0, 5], [7, 2, 5],
    [6, 4, 1], [3, 6, 1],
], dtype=np.int64)

_N_VERTS = len(_VERTS)   # 8
_N_FACES = len(_FACES)   # 12


def _make_python(verts, faces):
    """Build a Triangulation using the Python fallback path."""
    with unittest.mock.patch.object(tri_module, '_HAS_ACCEL', False):
        return Triangulation.from_arrays(verts, faces)


def _make_accel(verts, faces):
    """Build a Triangulation using the Cython accelerated path.

    Skips the test if the compiled extension is not available.
    """
    pytest.importorskip("pykarambola._accel")
    # _HAS_ACCEL is already True when the extension compiled successfully;
    # no patching needed.
    return Triangulation.from_arrays(verts, faces)


# ---------------------------------------------------------------------------
# Shared correctness tests (used by both TestPythonFallback and TestAccelPath)
# ---------------------------------------------------------------------------

class _TriangulationTests:
    """Correctness checks shared between the Python and Cython code paths.

    Subclasses must implement ``_make(verts, faces) -> Triangulation``.
    """

    def _make(self, verts, faces):
        raise NotImplementedError

    def test_neighbour_table_shape(self):
        t = self._make(_VERTS, _FACES)
        assert t._neighbours.shape == (_N_FACES, 3)

    def test_all_edges_assigned_on_closed_mesh(self):
        t = self._make(_VERTS, _FACES)
        assert np.all(t._neighbours != NEIGHBOUR_UNASSIGNED)

    def test_neighbour_relation_is_symmetric(self):
        t = self._make(_VERTS, _FACES)
        for i in range(_N_FACES):
            for j in range(3):
                k = t._neighbours[i, j]
                if k == NEIGHBOUR_UNASSIGNED:
                    continue
                # There must be some edge of triangle k that points back to i.
                assert i in t._neighbours[k], (
                    f"neighbour relation not symmetric: tri {i} edge {j} -> tri {k}, "
                    f"but {k}'s neighbours are {t._neighbours[k]}"
                )

    def test_vertex_triangles_total_count(self):
        t = self._make(_VERTS, _FACES)
        total = sum(len(t.get_triangles_of_vertex(v)) for v in range(_N_VERTS))
        assert total == 3 * _N_FACES

    def test_each_triangle_in_vertex_lists(self):
        t = self._make(_VERTS, _FACES)
        for i in range(_N_FACES):
            for j in range(3):
                v = int(_FACES[i, j])
                assert i in list(t.get_triangles_of_vertex(v))


# ---------------------------------------------------------------------------
# Python fallback
# ---------------------------------------------------------------------------

class TestPythonFallback(_TriangulationTests):
    """Correctness checks for the pure-Python fallback code path."""

    def _make(self, verts, faces):
        return _make_python(verts, faces)

    # This test pins internal storage details intentionally: it documents
    # that the Python path uses a list-of-lists layout rather than CSR.
    # It will break if the internal representation changes, which is acceptable.
    def test_uses_list_of_lists_storage(self):
        t = self._make(_VERTS, _FACES)
        assert t._vt_offsets is None
        assert t._vt_indices is None
        assert t._vertex_triangles is not None


# ---------------------------------------------------------------------------
# Cython accelerated path
# ---------------------------------------------------------------------------

class TestAccelPath(_TriangulationTests):
    """Correctness checks for the Cython accelerated code path.

    All tests in this class are skipped when the extension is not compiled.
    """

    def _make(self, verts, faces):
        return _make_accel(verts, faces)

    # This test pins internal storage details intentionally: it documents
    # that the Cython path uses CSR layout rather than a list-of-lists.
    # It will break if the internal representation changes, which is acceptable.
    def test_uses_csr_storage(self):
        t = self._make(_VERTS, _FACES)
        assert t._vt_offsets is not None
        assert t._vt_indices is not None
        assert t._vertex_triangles is None


# ---------------------------------------------------------------------------
# Agreement between both paths
# ---------------------------------------------------------------------------

class TestBothPathsAgree:
    """Both code paths must produce identical results on the same mesh.

    Skipped when the extension is not compiled.
    """

    def test_neighbour_tables_match(self):
        t_py = _make_python(_VERTS, _FACES)
        t_ac = _make_accel(_VERTS, _FACES)
        np.testing.assert_array_equal(t_py._neighbours, t_ac._neighbours)

    def test_vertex_triangle_lists_match(self):
        t_py = _make_python(_VERTS, _FACES)
        t_ac = _make_accel(_VERTS, _FACES)
        for v in range(_N_VERTS):
            py_tris = sorted(t_py.get_triangles_of_vertex(v))
            ac_tris = sorted(t_ac.get_triangles_of_vertex(v))
            assert py_tris == ac_tris, (
                f"vertex {v}: python={py_tris}, accel={ac_tris}"
            )
