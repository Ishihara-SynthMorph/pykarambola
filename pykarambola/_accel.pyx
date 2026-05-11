# cython: boundscheck=False, wraparound=False, cdivision=True
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Keisuke Ishihara, Yajushi Khurana
# This file is part of pykarambola, a Python port of karambola.
# See LICENSE for the full license text.
"""
Cython accelerator for triangulation initialization.

Provides fast implementations of:
- build_neighbour_table: edge-hash triangle neighbour lookup
- build_vertex_triangles: vertex -> triangle CSR arrays
"""

import numpy as np
cimport numpy as cnp

cnp.import_array()


def build_neighbour_table(cnp.int64_t[:, :] faces, int F, int neighbour_unassigned):
    """Build triangle neighbour table via edge hashing.

    Parameters
    ----------
    faces : (F, 3) int64 array
        Triangle vertex indices.
    F : int
        Number of triangles.
    neighbour_unassigned : int
        Sentinel value for unassigned neighbours.

    Returns
    -------
    neighbours : (F, 3) int64 ndarray
    """
    neighbours = np.full((F, 3), neighbour_unassigned, dtype=np.int64)
    cdef cnp.int64_t[:, :] nb_view = neighbours

    cdef dict edge_map = {}
    cdef int i, j
    cdef cnp.int64_t v0, v1
    cdef tuple key
    cdef int other_tri, other_edge

    for i in range(F):
        for j in range(3):
            v0 = faces[i, j]
            v1 = faces[i, (j + 1) % 3]
            if v0 < v1:
                key = (v0, v1)
            else:
                key = (v1, v0)

            if key in edge_map:
                other_tri, other_edge = edge_map[key]
                nb_view[i, j] = other_tri
                nb_view[other_tri, other_edge] = i
            else:
                edge_map[key] = (i, j)

    return np.asarray(nb_view)


def build_vertex_triangles(cnp.int64_t[:, :] faces, int F, int V):
    """Build vertex -> triangle CSR arrays.

    Parameters
    ----------
    faces : (F, 3) int64 array
        Triangle vertex indices.
    F : int
        Number of triangles.
    V : int
        Number of vertices.

    Returns
    -------
    offsets : (V+1,) int64 ndarray
        CSR offsets: triangles for vertex v are indices[offsets[v]:offsets[v+1]].
    indices : (3*F,) int64 ndarray
        CSR data array of triangle indices.
    """
    counts = np.zeros(V, dtype=np.int64)
    cdef cnp.int64_t[:] counts_view = counts
    cdef int i, j
    cdef cnp.int64_t vid

    # Pass 1: count triangles per vertex
    for i in range(F):
        for j in range(3):
            vid = faces[i, j]
            counts_view[vid] += 1

    # Build offsets from counts
    offsets = np.zeros(V + 1, dtype=np.int64)
    cdef cnp.int64_t[:] off_view = offsets
    for i in range(V):
        off_view[i + 1] = off_view[i] + counts_view[i]

    # Pass 2: fill indices
    cdef cnp.int64_t total = off_view[V]
    indices = np.empty(total, dtype=np.int64)
    cdef cnp.int64_t[:] idx_view = indices

    # Reset counts to use as current insertion position
    for i in range(V):
        counts_view[i] = 0

    for i in range(F):
        for j in range(3):
            vid = faces[i, j]
            idx_view[off_view[vid] + counts_view[vid]] = i
            counts_view[vid] += 1

    return np.asarray(off_view), np.asarray(idx_view)
