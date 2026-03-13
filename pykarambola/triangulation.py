# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Yajushi Khurana and Keisuke Ishihara
# This file is part of pykarambola, a Python port of karambola.
# See LICENSE for the full license text.
"""
Triangulation data structure for representing 3D triangulated surfaces.

Uses contiguous NumPy arrays for vectorized computation of Minkowski tensors.
"""

import warnings

import numpy as np

try:
    from ._accel import build_neighbour_table, build_vertex_triangles
    _HAS_ACCEL = True
except ImportError:
    _HAS_ACCEL = False

LABEL_UNASSIGNED = -300
NEIGHBOUR_UNASSIGNED = -200


class Triangulation:
    """Stores a triangulated surface mesh with vertices, triangles, and labels.

    Internally stores data as contiguous NumPy arrays and precomputes
    per-triangle geometry (normals, areas, centroids) and per-edge data
    (dihedral angles, edge lengths) for vectorized functional calculations.
    """

    @classmethod
    def from_arrays(cls, verts, faces, labels=None):
        """Construct a Triangulation from numpy arrays.

        Parameters
        ----------
        verts : (V, 3) array_like
            Vertex positions.
        faces : (F, 3) array_like
            Triangle vertex indices.
        labels : (F,) array_like or None
            Per-face labels. If None, all faces get label 0.

        Returns
        -------
        Triangulation
        """
        tri = cls()
        tri._verts = np.asarray(verts, dtype=np.float64).copy()
        tri._faces = np.asarray(faces, dtype=np.int64).copy()
        F = len(tri._faces)
        if labels is not None:
            tri._labels = np.asarray(labels, dtype=np.int64).copy()
        else:
            tri._labels = np.zeros(F, dtype=np.int64)
        tri._vertex_numbers = np.arange(len(tri._verts), dtype=np.int64)
        tri._finalized = False
        tri._build_vertex_polygon_lookup()
        tri._build_polygon_polygon_lookup()
        tri._precompute()
        return tri

    def __init__(self):
        # Append-mode lists (used by parse_poly_file)
        self._verts_list = []
        self._vertex_numbers_list = []
        self._faces_list = []
        self._labels_list = []

        # Array storage (set during finalization)
        self._verts = None           # (V, 3) float64
        self._faces = None           # (F, 3) int64
        self._labels = None          # (F,)   int64
        self._vertex_numbers = None  # (V,)   int64
        self._vertex_triangles = []  # list of lists: triangles per vertex (fallback)
        self._vt_offsets = None      # (V+1,) int64 CSR offsets (accel mode)
        self._vt_indices = None      # (3F,) int64 CSR data (accel mode)
        self._neighbours = None      # (F, 3) int64

        # Precomputed geometry
        self._normals = None         # (F, 3) float64
        self._areas = None           # (F,)   float64
        self._coms = None            # (F, 3) float64
        self._edge_lengths = None    # (F, 3) float64
        self._dihedral_angles = None # (F, 3) float64

        self._finalized = False

    # ------------------------------------------------------------------
    # Append-based construction (used by poly file parser)
    # ------------------------------------------------------------------

    def append_vertex(self, x, y, z, number_in_file=0):
        idx = len(self._verts_list)
        self._verts_list.append((x, y, z))
        self._vertex_numbers_list.append(number_in_file)
        return idx

    def change_pos_of_vertex(self, i, x, y, z):
        if self._verts is not None:
            self._verts[i] = [x, y, z]
            self._finalized = False
        else:
            self._verts_list[i] = (x, y, z)

    def append_triangle(self, v0, v1, v2, label=LABEL_UNASSIGNED):
        self._faces_list.append((v0, v1, v2))
        self._labels_list.append(label)

    def _consolidate_lists(self):
        """Convert append-mode lists to arrays."""
        if self._verts is None:
            if self._verts_list:
                self._verts = np.array(self._verts_list, dtype=np.float64)
            else:
                self._verts = np.empty((0, 3), dtype=np.float64)
            self._vertex_numbers = np.array(self._vertex_numbers_list, dtype=np.int64)
        if self._faces is None:
            if self._faces_list:
                self._faces = np.array(self._faces_list, dtype=np.int64)
            else:
                self._faces = np.empty((0, 3), dtype=np.int64)
            self._labels = np.array(self._labels_list, dtype=np.int64)

    # ------------------------------------------------------------------
    # Lookup table construction
    # ------------------------------------------------------------------

    def _build_vertex_polygon_lookup(self):
        """Build vertex -> triangle lookup using arrays."""
        V = len(self._verts)
        F = len(self._faces)
        if _HAS_ACCEL and F > 0:
            self._vt_offsets, self._vt_indices = build_vertex_triangles(
                self._faces, F, V)
            self._vertex_triangles = None  # CSR mode
        else:
            self._vt_offsets = None
            self._vt_indices = None
            self._vertex_triangles = [[] for _ in range(V)]
            for i in range(F):
                for j in range(3):
                    self._vertex_triangles[self._faces[i, j]].append(i)

    def _build_polygon_polygon_lookup(self):
        """Build triangle neighbour table using edge hashing."""
        F = len(self._faces)
        if _HAS_ACCEL and F > 0:
            self._neighbours = build_neighbour_table(
                self._faces, F, NEIGHBOUR_UNASSIGNED)
        else:
            self._neighbours = np.full((F, 3), NEIGHBOUR_UNASSIGNED, dtype=np.int64)
            edge_map = {}
            for i in range(F):
                for j in range(3):
                    v0 = self._faces[i, j]
                    v1 = self._faces[i, (j + 1) % 3]
                    key = (min(v0, v1), max(v0, v1))
                    if key in edge_map:
                        other_tri, other_edge = edge_map[key]
                        self._neighbours[i, j] = other_tri
                        self._neighbours[other_tri, other_edge] = i
                    else:
                        edge_map[key] = (i, j)

    def create_vertex_polygon_lookup_table(self):
        """Public interface — consolidates lists then builds lookup."""
        self._consolidate_lists()
        self._build_vertex_polygon_lookup()

    def create_polygon_polygon_lookup_table(self):
        """Public interface — consolidates lists then builds polygon lookup + precompute."""
        self._consolidate_lists()
        self._build_polygon_polygon_lookup()
        self._precompute()

    # ------------------------------------------------------------------
    # Precompute geometry
    # ------------------------------------------------------------------

    def _precompute(self):
        """Compute normals, areas, centroids, edge lengths, dihedral angles."""
        F = len(self._faces)

        # Vertices of each triangle: (F, 3)
        v0 = self._verts[self._faces[:, 0]]
        v1 = self._verts[self._faces[:, 1]]
        v2 = self._verts[self._faces[:, 2]]

        # Cross products and normals
        edge1 = v1 - v0
        edge2 = v2 - v0
        cross = np.cross(edge1, edge2)  # (F, 3)
        norms = np.linalg.norm(cross, axis=1)  # (F,)
        n_degenerate = int(np.sum(norms == 0))
        if n_degenerate:
            warnings.warn(
                f"{n_degenerate} degenerate (zero-area) triangle(s) detected; "
                "their normals are undefined and dihedral-angle-based quantities "
                "(w200, w202, w210, w220) may be unreliable.",
                stacklevel=3,
            )
        # Avoid division by zero for degenerate triangles
        safe_norms = np.where(norms > 0, norms, 1.0)
        self._normals = cross / safe_norms[:, None]
        self._areas = norms / 2.0
        self._coms = (v0 + v1 + v2) / 3.0

        # Edge lengths: edge j connects vertex j to vertex (j+1)%3
        self._edge_lengths = np.empty((F, 3), dtype=np.float64)
        self._edge_lengths[:, 0] = np.linalg.norm(v1 - v0, axis=1)
        self._edge_lengths[:, 1] = np.linalg.norm(v2 - v1, axis=1)
        self._edge_lengths[:, 2] = np.linalg.norm(v0 - v2, axis=1)

        # Dihedral angles
        self._dihedral_angles = np.zeros((F, 3), dtype=np.float64)
        for j in range(3):
            nb = self._neighbours[:, j]  # (F,)
            mask = nb != NEIGHBOUR_UNASSIGNED
            if not np.any(mask):
                continue
            tri_idx = np.where(mask)[0]
            nb_idx = nb[tri_idx]

            n1 = self._normals[tri_idx]
            n2 = self._normals[nb_idx]
            cos_angle = np.sum(n1 * n2, axis=1)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            alpha = np.arccos(cos_angle)

            # Convexity check
            com1 = self._coms[tri_idx]
            com2 = self._coms[nb_idx]
            convex_vec = com1 + n1 - com2 - n2
            concave_vec = com1 - n1 - com2 + n2
            convex_sq = np.sum(convex_vec ** 2, axis=1)
            concave_sq = np.sum(concave_vec ** 2, axis=1)
            sign = np.where(convex_sq < concave_sq, -1.0, 1.0)
            self._dihedral_angles[tri_idx, j] = alpha * sign

        # Precompute vertex angles: angle at vertex j of triangle i
        self._vertex_angles = np.empty((F, 3), dtype=np.float64)
        for j in range(3):
            vi = self._verts[self._faces[:, j]]
            vj = self._verts[self._faces[:, (j + 1) % 3]]
            vk = self._verts[self._faces[:, (j + 2) % 3]]
            a_vec = vi - vj
            b_vec = vi - vk
            a_norm = np.linalg.norm(a_vec, axis=1)
            b_norm = np.linalg.norm(b_vec, axis=1)
            safe_denom = np.where((a_norm * b_norm) > 0, a_norm * b_norm, 1.0)
            cos_val = np.sum(a_vec * b_vec, axis=1) / safe_denom
            cos_val = np.clip(cos_val, -1.0, 1.0)
            self._vertex_angles[:, j] = np.arccos(cos_val)

        # Precompute per-vertex angle sums
        V = len(self._verts)
        self._vertex_angle_sums = np.zeros(V, dtype=np.float64)
        for j in range(3):
            np.add.at(self._vertex_angle_sums, self._faces[:, j], self._vertex_angles[:, j])

        self._finalized = True

    # ------------------------------------------------------------------
    # Accessor methods (backward compatible, index into arrays)
    # ------------------------------------------------------------------

    def n_triangles(self):
        if self._faces is not None:
            return len(self._faces)
        return len(self._faces_list)

    def n_vertices(self):
        if self._verts is not None:
            return len(self._verts)
        return len(self._verts_list)

    def ith_vertex_of_triangle(self, a, i):
        if self._faces is not None:
            return int(self._faces[a, i])
        return self._faces_list[a][i]

    def label_of_triangle(self, i):
        if self._labels is not None:
            return int(self._labels[i])
        return self._labels_list[i]

    def get_pos_of_vertex(self, a):
        if self._verts is not None:
            return self._verts[a]
        return np.array(self._verts_list[a], dtype=np.float64)

    def get_original_number_of_vertex_in_file(self, a):
        if self._vertex_numbers is not None:
            return int(self._vertex_numbers[a])
        return self._vertex_numbers_list[a]

    def get_triangles_of_vertex(self, a):
        if self._vt_offsets is not None:
            return self._vt_indices[self._vt_offsets[a]:self._vt_offsets[a + 1]]
        return self._vertex_triangles[a]

    def ith_neighbour_of_triangle(self, a, i):
        return int(self._neighbours[a, i])

    def area_of_triangle(self, i):
        return float(self._areas[i])

    def normal_vector_of_triangle(self, i):
        return self._normals[i]

    def com_of_triangle(self, i):
        return self._coms[i]

    def get_edge_length(self, a, i):
        return float(self._edge_lengths[a, i])

    def get_angle_of_ith_edge(self, a, i):
        return float(self._vertex_angles[a, i])

    def sum_of_angles_of_ith_vertex(self, i):
        return float(self._vertex_angle_sums[i])
