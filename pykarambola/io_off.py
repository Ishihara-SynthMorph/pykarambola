# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Keisuke Ishihara, Yajushi Khurana
# This file is part of pykarambola, a Python port of karambola.
# See LICENSE for the full license text.
"""
Parser for .off file format (Object File Format).
"""

from .triangulation import Triangulation, LABEL_UNASSIGNED


def parse_off_file(filepath, with_labels=False):
    """Parse an .off file and return a Triangulation.

    Parameters
    ----------
    filepath : str or path-like
        Path to the .off file.
    with_labels : bool
        If True, extract labels from the alpha channel of facet colors.

    Returns
    -------
    Triangulation

    Examples
    --------
    >>> import pykarambola as pk
    >>> tri = pk.parse_off_file("my_surface.off")
    >>> result = pk.minkowski_tensors(tri)
    >>> result["w000"]   # enclosed volume
    """
    tri = Triangulation()

    with open(filepath, "r") as f:
        lines = f.readlines()

    # Filter comments and blank lines, normalize
    cleaned = []
    for line in lines:
        line = line.strip()
        if line.startswith("#") or line == "":
            continue
        # Strip inline comments
        if "#" in line:
            line = line[:line.index("#")].strip()
        cleaned.append(line)

    if not cleaned:
        raise ValueError("Empty .off file")

    idx = 0

    # Expect "OFF" header
    if cleaned[idx].strip() != "OFF":
        raise ValueError(f"Expected OFF header, got {cleaned[idx]}")
    idx += 1

    # Read counts: num_vertices num_facets num_edges
    parts = cleaned[idx].split()
    num_vertices = int(parts[0])
    num_facets = int(parts[1])
    # num_edges = int(parts[2])  # ignored
    idx += 1

    # Read vertices
    vertex_map = {}
    for v_idx in range(num_vertices):
        parts = cleaned[idx].split()
        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
        vert_id = tri.append_vertex(x, y, z, v_idx)
        vertex_map[v_idx] = vert_id
        idx += 1

    # Read facets
    for _ in range(num_facets):
        parts = cleaned[idx].split()
        n_verts = int(parts[0])
        vertex_ids = []
        for vi in range(1, n_verts + 1):
            v = int(parts[vi])
            vertex_ids.append(vertex_map[v])

        label = LABEL_UNASSIGNED
        # Check for color data (r g b [alpha]) after vertex indices
        remaining = parts[1 + n_verts:]
        if with_labels and len(remaining) >= 4:
            # r, g, b, alpha
            try:
                alpha = int(remaining[3])
                label = alpha
            except (ValueError, IndexError):
                pass

        # Fan triangulation
        if len(vertex_ids) > 3:
            for i in range(1, len(vertex_ids) - 1):
                tri.append_triangle(vertex_ids[0], vertex_ids[i], vertex_ids[i + 1], label)
        elif len(vertex_ids) == 3:
            tri.append_triangle(vertex_ids[0], vertex_ids[1], vertex_ids[2], label)

        idx += 1

    return tri


def is_off_file(filename):
    """Check if the filename indicates an .off file."""
    return filename.lower().endswith(".off")
