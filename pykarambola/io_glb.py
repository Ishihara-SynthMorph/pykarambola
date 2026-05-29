# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Keisuke Ishihara, Yajushi Khurana
# This file is part of pykarambola, a Python port of karambola.
# See LICENSE for the full license text.
"""
Parser for .glb/.gltf file formats (Binary glTF / glTF).
"""

from .triangulation import Triangulation, LABEL_UNASSIGNED


def parse_glb_file(filepath, with_labels=False):
    """Parse a .glb or .gltf file and return a Triangulation.

    Requires the ``trimesh`` package.

    Parameters
    ----------
    filepath : str or path-like
        Path to the .glb or .gltf file.
    with_labels : bool
        If True, assign integer labels per sub-mesh in a Scene.

    Returns
    -------
    Triangulation

    Examples
    --------
    >>> import pykarambola as pk
    >>> tri = pk.parse_glb_file("my_surface.glb")  # requires trimesh
    >>> result = pk.minkowski_tensors(tri)
    >>> result["w000"]   # enclosed volume
    """
    try:
        import trimesh
    except ImportError:
        raise ImportError(
            "trimesh is required for GLB support: pip install trimesh"
        )

    tri = Triangulation()
    data = trimesh.load(filepath)

    meshes = []
    if isinstance(data, trimesh.Scene):
        for name, geom in data.geometry.items():
            if isinstance(geom, trimesh.Trimesh):
                meshes.append(geom)
    elif isinstance(data, trimesh.Trimesh):
        meshes.append(data)
    else:
        raise ValueError(f"Unsupported trimesh type: {type(data)}")

    vertex_offset = 0
    for mesh_idx, mesh in enumerate(meshes):
        label = mesh_idx if with_labels else LABEL_UNASSIGNED

        vertex_map = {}
        for i, v in enumerate(mesh.vertices):
            vert_id = tri.append_vertex(
                float(v[0]), float(v[1]), float(v[2]), vertex_offset + i
            )
            vertex_map[i] = vert_id

        for face in mesh.faces:
            vertex_ids = [vertex_map[int(vi)] for vi in face]

            # Fan triangulation (trimesh usually pre-triangulates)
            if len(vertex_ids) > 3:
                for i in range(1, len(vertex_ids) - 1):
                    tri.append_triangle(
                        vertex_ids[0], vertex_ids[i], vertex_ids[i + 1], label
                    )
            elif len(vertex_ids) == 3:
                tri.append_triangle(
                    vertex_ids[0], vertex_ids[1], vertex_ids[2], label
                )

        vertex_offset += len(mesh.vertices)

    return tri


def is_glb_file(filename):
    """Check if the filename indicates a .glb or .gltf file."""
    lower = filename.lower()
    return lower.endswith(".glb") or lower.endswith(".gltf")
