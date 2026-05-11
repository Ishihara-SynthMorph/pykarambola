# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Keisuke Ishihara, Yajushi Khurana
# This file is part of pykarambola, a Python port of karambola.
# See LICENSE for the full license text.
"""
Surface validation and statistics.
"""

import numpy as np
from .triangulation import NEIGHBOUR_UNASSIGNED
from .results import SurfaceStatistics


def check_surface(calc_options, surface):
    """Check surface properties (closure status, statistics, validation).

    Parameters
    ----------
    calc_options : CalcOptions
        Will be updated with label closure status.
    surface : Triangulation
        The surface to check.

    Returns
    -------
    SurfaceStatistics
    """
    stats = SurfaceStatistics()

    if surface.n_triangles() == 0:
        raise RuntimeError("There are no polygons in your .poly file")

    # Initialize all labels as closed (status 0)
    for lab in np.unique(surface._labels):
        calc_options.create_label(int(lab), 0)

    # Check for shared edges (different labels on neighbours) -> status 1
    nb = surface._neighbours  # (F, 3)
    labels = surface._labels  # (F,)
    mask_valid = nb != NEIGHBOUR_UNASSIGNED  # (F, 3)
    # For each valid neighbour, check if labels differ
    nb_clamped = np.where(mask_valid, nb, 0)  # safe indices for gather
    nb_labels = labels[nb_clamped]  # (F, 3) — labels of neighbours
    shared_edge = mask_valid & (labels[:, None] != nb_labels)  # (F, 3)
    tri_with_shared = np.any(shared_edge, axis=1)
    for lab in np.unique(labels[tri_with_shared]):
        calc_options.create_label(int(lab), 1)

    # Check for open edges (no neighbour) -> status 2
    open_edge = np.any(nb == NEIGHBOUR_UNASSIGNED, axis=1)  # (F,)
    for lab in np.unique(labels[open_edge]):
        calc_options.create_label(int(lab), 2)

    # Edge length statistics
    stats.shortest_edge = float(np.min(surface._edge_lengths))
    stats.longest_edge = float(np.max(surface._edge_lengths))

    # Area statistics
    stats.smallest_area = float(np.min(surface._areas))
    stats.largest_area = float(np.max(surface._areas))

    if stats.largest_area > 0 and stats.smallest_area / stats.largest_area < 1e-12:
        print(f"\nWARNING: there is something wrong with the area of your facets!")
        print(f"smallest area: {stats.smallest_area}")
        print(f"largest area:  {stats.largest_area}\n")

    # Check for multiple objects at one vertex and normal consistency
    for i in range(surface.n_vertices()):
        tris = surface.get_triangles_of_vertex(i)
        if len(tris) == 0:
            continue

        sum_of_triangles = 1
        start_triangle = tris[0]
        old_triangle = NEIGHBOUR_UNASSIGNED
        triangle = tris[0]
        new_triangle = tris[0]

        neigh_un = False
        normal_wrong = False

        for _ in range(len(tris)):
            triangle = new_triangle
            vertex_id = 0
            for k in range(3):
                if surface.ith_vertex_of_triangle(triangle, k) == i:
                    vertex_id = k

            if surface.ith_neighbour_of_triangle(triangle, vertex_id) == NEIGHBOUR_UNASSIGNED:
                neigh_un = True
                break

            new_triangle = surface.ith_neighbour_of_triangle(triangle, vertex_id)

            if old_triangle == new_triangle:
                normal_wrong = True
                break

            if new_triangle == start_triangle:
                break

            sum_of_triangles += 1
            old_triangle = triangle

        if normal_wrong:
            raise RuntimeError(
                f"shortest edge = {stats.shortest_edge}\n"
                f"longest edge  = {stats.longest_edge}\n"
                f"smallest area = {stats.smallest_area}\n"
                f"largest area  = {stats.largest_area}\n"
                f"your polyfile is damaged\n"
                f"your normals are messed up!!\n"
                f"error occurs at vertex: {surface.get_original_number_of_vertex_in_file(i)}"
            )

        if len(tris) != sum_of_triangles and not neigh_un:
            raise RuntimeError(
                f"your polyfile is damaged\n"
                f"there are more than one objects at vertex "
                f"{surface.get_original_number_of_vertex_in_file(i)}: "
                f"{surface.get_pos_of_vertex(i)}"
            )

    return stats
