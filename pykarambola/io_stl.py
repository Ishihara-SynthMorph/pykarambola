# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2026 Keisuke Ishihara, Yajushi Khurana
# This file is part of pykarambola, a Python port of karambola.
# See LICENSE for the full license text.
"""
Parser for .stl file format (ASCII and binary STereoLithography).
"""

from .triangulation import Triangulation, LABEL_UNASSIGNED


def parse_stl_file(filepath):
    """Parse an ASCII or binary STL file and return a Triangulation.

    Requires the ``numpy-stl`` package (``pip install "pykarambola[stl]"``).

    Vertex deduplication is performed automatically so that coincident vertices
    are merged into a proper shared-vertex indexed mesh.

    Parameters
    ----------
    filepath : str or path-like
        Path to the .stl file.

    Returns
    -------
    Triangulation

    Raises
    ------
    ImportError
        If ``numpy-stl`` is not installed.
    ValueError
        If the file is malformed or cannot be parsed.

    Examples
    --------
    >>> import pykarambola as pk
    >>> tri = pk.parse_stl_file("mesh.stl")  # requires numpy-stl
    >>> result = pk.minkowski_tensors(tri)
    >>> result["w000"]   # enclosed volume
    """
    try:
        from stl import mesh as stl_mesh
    except ImportError:
        raise ImportError(
            "numpy-stl is required for STL support: "
            'pip install "pykarambola[stl]"'
        )

    try:
        data = stl_mesh.Mesh.from_file(str(filepath))
    except Exception as exc:
        raise ValueError(f"Failed to parse STL file '{filepath}': {exc}") from exc

    if len(data.vectors) == 0:
        raise ValueError(
            f"Failed to parse STL file '{filepath}': no faces found (file may be empty or malformed)"
        )

    # STL stores three vertices per face with no shared-vertex indexing.
    # Deduplicate by mapping (x, y, z) tuples to vertex IDs.
    tri = Triangulation()
    vertex_index = {}  # (x, y, z) -> vertex id in Triangulation

    for face in data.vectors:
        face_vertex_ids = []
        for v in face:
            key = (float(v[0]), float(v[1]), float(v[2]))
            if key not in vertex_index:
                vid = tri.append_vertex(key[0], key[1], key[2], len(vertex_index))
                vertex_index[key] = vid
            face_vertex_ids.append(vertex_index[key])

        tri.append_triangle(
            face_vertex_ids[0], face_vertex_ids[1], face_vertex_ids[2],
            LABEL_UNASSIGNED,
        )

    return tri


def is_stl_file(filename):
    """Check if the filename indicates a .stl file."""
    return filename.lower().endswith(".stl")
