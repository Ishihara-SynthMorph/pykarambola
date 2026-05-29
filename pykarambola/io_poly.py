# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Keisuke Ishihara, Yajushi Khurana
# This file is part of pykarambola, a Python port of karambola.
# See LICENSE for the full license text.
"""
Parser for .poly file format (Geomview polyhedron format).
"""

import re
from .triangulation import Triangulation, LABEL_UNASSIGNED


def _label_from_alpha_value(props):
    """Extract label from color attribute alpha value in properties list."""
    for prop in props:
        if prop.startswith("c("):
            # Find last comma, extract alpha value
            idx = prop.rfind(",")
            if idx == -1:
                raise ValueError(f"Malformed color attribute: {prop}")
            alpha_str = prop[idx + 1:].strip()
            # Remove trailing ')'
            alpha_str = alpha_str.rstrip(")")
            return int(alpha_str)
    raise ValueError("No color attribute found for label extraction")


def parse_poly_file(filepath, with_labels=False):
    """Parse a .poly file and return a Triangulation.

    Parameters
    ----------
    filepath : str or path-like
        Path to the .poly file.
    with_labels : bool
        If True, extract labels from the color alpha channel.

    Returns
    -------
    Triangulation

    Examples
    --------
    >>> import pykarambola as pk
    >>> tri = pk.parse_poly_file("my_surface.poly")
    >>> result = pk.minkowski_tensors(tri)
    >>> result["w000"]   # enclosed volume
    """
    tri = Triangulation()

    with open(filepath, "r") as f:
        content = f.read()

    # Normalize line endings
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    lines = content.split("\n")

    line_idx = 0

    def skip_blank_and_comments():
        nonlocal line_idx
        while line_idx < len(lines):
            stripped = lines[line_idx].strip()
            if stripped == "" or stripped.startswith("#"):
                line_idx += 1
            else:
                break

    # Expect "POINTS"
    skip_blank_and_comments()
    if lines[line_idx].strip() != "POINTS":
        raise ValueError(f"Expected POINTS, got {lines[line_idx].strip()}")
    line_idx += 1

    # vertex_map: maps original vertex number -> internal index
    vertex_map = {}

    # Read vertices
    while line_idx < len(lines):
        skip_blank_and_comments()
        if line_idx >= len(lines):
            break
        stripped = lines[line_idx].strip()
        if stripped == "POLYS":
            break
        if not stripped or not stripped[0].isdigit():
            break

        # Parse "number: x y z [properties...]"
        colon_idx = stripped.index(":")
        number = int(stripped[:colon_idx].strip())
        rest = stripped[colon_idx + 1:].strip()

        # Parse coordinates and properties
        tokens = _tokenize_with_parens(rest)
        if len(tokens) < 3:
            raise ValueError(
                f"Expected at least 3 coordinates for vertex {number}, got {len(tokens)}"
            )

        x, y, z = float(tokens[0]), float(tokens[1]), float(tokens[2])
        vert_id = tri.append_vertex(x, y, z, number)
        vertex_map[number] = vert_id
        line_idx += 1

    # Expect "POLYS"
    skip_blank_and_comments()
    if lines[line_idx].strip() != "POLYS":
        raise ValueError(f"Expected POLYS, got {lines[line_idx].strip()}")
    line_idx += 1

    # Read facets
    while line_idx < len(lines):
        skip_blank_and_comments()
        if line_idx >= len(lines):
            break
        stripped = lines[line_idx].strip()
        if stripped == "END":
            break
        if not stripped or not stripped[0].isdigit():
            break

        # Parse "number: v1 v2 v3 ... [properties...]"
        colon_idx = stripped.index(":")
        rest = stripped[colon_idx + 1:].strip()

        tokens = _tokenize_with_parens(rest)

        # Separate vertex indices from properties
        vertex_ids = []
        properties = []
        for token in tokens:
            if token[0].isdigit() and not vertex_ids or (
                vertex_ids and token[0].isdigit() and "(" not in token and "<" not in token
            ):
                try:
                    v = int(token)
                    if v in vertex_map:
                        vertex_ids.append(vertex_map[v])
                    else:
                        raise ValueError(f"Unknown vertex reference: {v}")
                except ValueError:
                    if "(" in token or "<" in token or not token[0].isdigit():
                        properties.append(token)
                    else:
                        raise
            else:
                properties.append(token)

        label = LABEL_UNASSIGNED
        if with_labels:
            try:
                label = _label_from_alpha_value(properties)
            except ValueError:
                label = LABEL_UNASSIGNED

        # Fan triangulation for polygons with >3 vertices
        if len(vertex_ids) > 3:
            for i in range(1, len(vertex_ids) - 1):
                tri.append_triangle(vertex_ids[0], vertex_ids[i], vertex_ids[i + 1], label)
        elif len(vertex_ids) == 3:
            tri.append_triangle(vertex_ids[0], vertex_ids[1], vertex_ids[2], label)

        line_idx += 1

    return tri


def _tokenize_with_parens(s):
    """Tokenize a string respecting parentheses grouping."""
    tokens = []
    current = ""
    paren_depth = 0
    for ch in s:
        if ch == "(":
            paren_depth += 1
            current += ch
        elif ch == ")":
            paren_depth -= 1
            current += ch
        elif ch in (" ", "\t") and paren_depth == 0:
            if current:
                tokens.append(current)
                current = ""
        else:
            current += ch
    if current:
        tokens.append(current)
    return tokens
