# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Keisuke Ishihara, Yajushi Khurana
# This file is part of pykarambola, a Python port of karambola.
# See LICENSE for the full license text.
"""
Parser for .obj file format (Wavefront OBJ).
"""

from .triangulation import Triangulation, LABEL_UNASSIGNED


def parse_obj_file(filepath, with_labels=False):
    """Parse an .obj file and return a Triangulation.

    Parameters
    ----------
    filepath : str or path-like
        Path to the .obj file.
    with_labels : bool
        If True, assign integer labels based on usemtl/g group names.

    Returns
    -------
    Triangulation

    Examples
    --------
    >>> import pykarambola as pk
    >>> tri = pk.parse_obj_file("my_surface.obj")
    >>> result = pk.minkowski_tensors(tri)
    >>> result["w000"]   # enclosed volume
    """
    tri = Triangulation()

    with open(filepath, "r") as f:
        lines = f.readlines()

    vertex_map = {}
    vertex_count = 0

    # Label tracking
    current_label = LABEL_UNASSIGNED
    group_label_map = {}
    next_label = 0

    for line in lines:
        line = line.strip()
        if line == "" or line.startswith("#"):
            continue

        parts = line.split()
        keyword = parts[0]

        if keyword == "v" and len(parts) >= 4:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            vert_id = tri.append_vertex(x, y, z, vertex_count)
            vertex_map[vertex_count] = vert_id
            vertex_count += 1

        elif keyword in ("usemtl", "g") and with_labels:
            group_name = parts[1] if len(parts) > 1 else ""
            if group_name not in group_label_map:
                group_label_map[group_name] = next_label
                next_label += 1
            current_label = group_label_map[group_name]

        elif keyword == "f":
            # Parse face vertex indices, handling v, v/vt, v/vt/vn, v//vn
            vertex_ids = []
            for token in parts[1:]:
                # Take only the vertex index (before first '/')
                idx_str = token.split("/")[0]
                idx = int(idx_str)
                # OBJ uses 1-based indices; negative indices are relative
                if idx > 0:
                    idx -= 1  # convert to 0-based
                else:
                    idx = vertex_count + idx  # negative index
                vertex_ids.append(vertex_map[idx])

            label = current_label if with_labels else LABEL_UNASSIGNED

            # Fan triangulation
            if len(vertex_ids) > 3:
                for i in range(1, len(vertex_ids) - 1):
                    tri.append_triangle(vertex_ids[0], vertex_ids[i], vertex_ids[i + 1], label)
            elif len(vertex_ids) == 3:
                tri.append_triangle(vertex_ids[0], vertex_ids[1], vertex_ids[2], label)

    return tri


def is_obj_file(filename):
    """Check if the filename indicates an .obj file."""
    return filename.lower().endswith(".obj")
