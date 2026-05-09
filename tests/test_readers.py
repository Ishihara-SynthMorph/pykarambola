"""
Tests for .poly, .off, .obj, and .glb file readers.
"""

import os
import tempfile
import pytest
import numpy as np

from pykarambola.io_poly import parse_poly_file
from pykarambola.io_off import parse_off_file, is_off_file
from pykarambola.io_obj import parse_obj_file, is_obj_file
from pykarambola.io_glb import parse_glb_file, is_glb_file
from pykarambola.io_stl import parse_stl_file, is_stl_file
from pykarambola.triangulation import LABEL_UNASSIGNED
from pykarambola.surface import check_surface
from pykarambola.results import CalcOptions
from pykarambola.minkowski import calculate_w000, calculate_w100

TEST_INPUTS = os.path.join(os.path.dirname(__file__), "fixtures")


class TestPolyReader:
    """Tests for .poly file parsing."""

    def test_box_poly(self):
        """Read box.poly and verify vertex/triangle counts."""
        filepath = os.path.join(TEST_INPUTS, "box_a=2_b=3_c=4.poly")
        surface = parse_poly_file(filepath)
        assert surface.n_vertices() == 8
        assert surface.n_triangles() == 12

    def test_box_poly_volume(self):
        """Read box.poly and verify volume computation."""
        filepath = os.path.join(TEST_INPUTS, "box_a=2_b=3_c=4.poly")
        surface = parse_poly_file(filepath)
        surface.create_vertex_polygon_lookup_table()
        surface.create_polygon_polygon_lookup_table()
        w000 = calculate_w000(surface)
        assert w000[LABEL_UNASSIGNED].result == pytest.approx(24.0, rel=1e-4)

    def test_empty_poly(self):
        """Empty poly should raise an error during surface check."""
        filepath = os.path.join(TEST_INPUTS, "empty.poly")
        surface = parse_poly_file(filepath)
        surface.create_vertex_polygon_lookup_table()
        surface.create_polygon_polygon_lookup_table()
        co = CalcOptions()
        with pytest.raises(RuntimeError, match="no polygons"):
            check_surface(co, surface)


class TestOffReader:
    """Tests for .off file parsing."""

    def test_cuboid_off(self):
        """Read cuboid.off and verify vertex/triangle counts."""
        filepath = os.path.join(TEST_INPUTS, "cuboid.off")
        surface = parse_off_file(filepath)
        assert surface.n_vertices() == 8
        # 6 faces * 2 triangles each = 12 triangles (quads are triangulated)
        assert surface.n_triangles() == 12

    def test_cuboid_off_volume(self):
        """Read cuboid.off and verify volume."""
        filepath = os.path.join(TEST_INPUTS, "cuboid.off")
        surface = parse_off_file(filepath)
        surface.create_vertex_polygon_lookup_table()
        surface.create_polygon_polygon_lookup_table()
        w000 = calculate_w000(surface)
        # Cuboid volume should be positive
        assert w000[LABEL_UNASSIGNED].result > 0

    def test_cuboid_with_labels(self):
        """Read cuboid with labels from alpha channel."""
        filepath = os.path.join(TEST_INPUTS, "cuboid-labels.off")
        if not os.path.exists(filepath):
            pytest.skip("cuboid-labels.off not found")
        surface = parse_off_file(filepath, with_labels=True)
        assert surface.n_vertices() >= 8

    def test_cuboid_with_comments(self):
        """Read cuboid with comments in .off file."""
        filepath = os.path.join(TEST_INPUTS, "cuboid-labelsWithComments.off")
        if not os.path.exists(filepath):
            pytest.skip("cuboid-labelsWithComments.off not found")
        surface = parse_off_file(filepath)
        assert surface.n_vertices() >= 8

    def test_is_off_file(self):
        assert is_off_file("test.off")
        assert is_off_file("test.OFF")
        assert not is_off_file("test.poly")
        assert not is_off_file("test.txt")


class TestObjReader:
    """Tests for .obj file parsing."""

    def test_box_obj(self):
        """Read box.obj and verify vertex/triangle counts."""
        filepath = os.path.join(TEST_INPUTS, "box.obj")
        surface = parse_obj_file(filepath)
        assert surface.n_vertices() == 8
        assert surface.n_triangles() == 12

    def test_box_obj_volume(self):
        """Read box.obj and verify volume = 24."""
        filepath = os.path.join(TEST_INPUTS, "box.obj")
        surface = parse_obj_file(filepath)
        surface.create_vertex_polygon_lookup_table()
        surface.create_polygon_polygon_lookup_table()
        w000 = calculate_w000(surface)
        assert w000[LABEL_UNASSIGNED].result == pytest.approx(24.0, rel=1e-4)

    def test_is_obj_file(self):
        assert is_obj_file("test.obj")
        assert is_obj_file("test.OBJ")
        assert not is_obj_file("test.poly")
        assert not is_obj_file("test.off")


class TestGlbReader:
    """Tests for .glb file parsing."""

    @pytest.fixture
    def box_glb(self, tmp_path):
        """Create a GLB fixture of the same box via trimesh."""
        trimesh = pytest.importorskip("trimesh")
        vertices = np.array([
            [ 1.0, -1.5,  2.0],
            [-1.0, -1.5,  2.0],
            [ 1.0,  1.5,  2.0],
            [-1.0,  1.5,  2.0],
            [-1.0, -1.5, -2.0],
            [ 1.0, -1.5, -2.0],
            [-1.0,  1.5, -2.0],
            [ 1.0,  1.5, -2.0],
        ])
        faces = np.array([
            [3, 1, 0], [2, 3, 0], [7, 5, 4], [6, 7, 4],
            [2, 7, 6], [3, 2, 6], [1, 4, 5], [0, 1, 5],
            [2, 0, 5], [7, 2, 5], [6, 4, 1], [3, 6, 1],
        ])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        filepath = str(tmp_path / "box.glb")
        mesh.export(filepath)
        return filepath

    def test_box_glb(self, box_glb):
        """Read GLB box and verify vertex/triangle counts."""
        surface = parse_glb_file(box_glb)
        assert surface.n_vertices() == 8
        assert surface.n_triangles() == 12

    def test_box_glb_volume(self, box_glb):
        """Read GLB box and verify volume = 24."""
        surface = parse_glb_file(box_glb)
        surface.create_vertex_polygon_lookup_table()
        surface.create_polygon_polygon_lookup_table()
        w000 = calculate_w000(surface)
        assert w000[LABEL_UNASSIGNED].result == pytest.approx(24.0, rel=1e-4)

    def test_is_glb_file(self):
        assert is_glb_file("test.glb")
        assert is_glb_file("test.gltf")
        assert is_glb_file("test.GLB")
        assert not is_glb_file("test.obj")
        assert not is_glb_file("test.off")


class TestStlReader:
    """Tests for .stl file parsing (ASCII and binary)."""

    # Box vertices and faces shared by both ASCII and binary fixtures.
    _vertices = np.array([
        [ 1.0, -1.5,  2.0],
        [-1.0, -1.5,  2.0],
        [ 1.0,  1.5,  2.0],
        [-1.0,  1.5,  2.0],
        [-1.0, -1.5, -2.0],
        [ 1.0, -1.5, -2.0],
        [-1.0,  1.5, -2.0],
        [ 1.0,  1.5, -2.0],
    ])
    _faces = np.array([
        [3, 1, 0], [2, 3, 0], [7, 5, 4], [6, 7, 4],
        [2, 7, 6], [3, 2, 6], [1, 4, 5], [0, 1, 5],
        [2, 0, 5], [7, 2, 5], [6, 4, 1], [3, 6, 1],
    ])

    def _write_ascii_stl(self, filepath):
        """Write a minimal ASCII STL file for a box."""
        verts = self._vertices
        faces = self._faces
        with open(filepath, "w") as f:
            f.write("solid box\n")
            for face in faces:
                v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
                f.write("  facet normal 0 0 0\n")
                f.write("    outer loop\n")
                f.write(f"      vertex {v0[0]} {v0[1]} {v0[2]}\n")
                f.write(f"      vertex {v1[0]} {v1[1]} {v1[2]}\n")
                f.write(f"      vertex {v2[0]} {v2[1]} {v2[2]}\n")
                f.write("    endloop\n")
                f.write("  endfacet\n")
            f.write("endsolid box\n")

    def _write_binary_stl(self, filepath):
        """Write a minimal binary STL file for a box."""
        import struct
        verts = self._vertices
        faces = self._faces
        with open(filepath, "wb") as f:
            f.write(b"\x00" * 80)  # 80-byte header
            f.write(struct.pack("<I", len(faces)))
            for face in faces:
                v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
                f.write(struct.pack("<fff", 0.0, 0.0, 0.0))  # normal
                f.write(struct.pack("<fff", *v0))
                f.write(struct.pack("<fff", *v1))
                f.write(struct.pack("<fff", *v2))
                f.write(struct.pack("<H", 0))  # attribute byte count

    @pytest.fixture
    def ascii_stl(self, tmp_path):
        pytest.importorskip("stl")
        filepath = str(tmp_path / "box_ascii.stl")
        self._write_ascii_stl(filepath)
        return filepath

    @pytest.fixture
    def binary_stl(self, tmp_path):
        pytest.importorskip("stl")
        filepath = str(tmp_path / "box_binary.stl")
        self._write_binary_stl(filepath)
        return filepath

    def test_ascii_stl_counts(self, ascii_stl):
        """Parse ASCII STL and verify vertex/triangle counts after dedup."""
        surface = parse_stl_file(ascii_stl)
        assert surface.n_vertices() == 8
        assert surface.n_triangles() == 12

    def test_ascii_stl_volume(self, ascii_stl):
        """Parse ASCII STL and verify volume = 24."""
        surface = parse_stl_file(ascii_stl)
        surface.create_vertex_polygon_lookup_table()
        surface.create_polygon_polygon_lookup_table()
        w000 = calculate_w000(surface)
        assert w000[LABEL_UNASSIGNED].result == pytest.approx(24.0, rel=1e-4)

    def test_binary_stl_counts(self, binary_stl):
        """Parse binary STL and verify vertex/triangle counts after dedup."""
        surface = parse_stl_file(binary_stl)
        assert surface.n_vertices() == 8
        assert surface.n_triangles() == 12

    def test_binary_stl_volume(self, binary_stl):
        """Parse binary STL and verify volume = 24."""
        surface = parse_stl_file(binary_stl)
        surface.create_vertex_polygon_lookup_table()
        surface.create_polygon_polygon_lookup_table()
        w000 = calculate_w000(surface)
        assert w000[LABEL_UNASSIGNED].result == pytest.approx(24.0, rel=1e-4)

    def test_importerror_without_numpy_stl(self, tmp_path, monkeypatch):
        """ImportError raised with install hint if numpy-stl is missing."""
        import sys
        monkeypatch.setitem(sys.modules, "stl", None)
        with pytest.raises(ImportError, match="numpy-stl"):
            parse_stl_file(str(tmp_path / "dummy.stl"))

    def test_malformed_stl_raises(self, tmp_path):
        """Malformed STL file raises ValueError."""
        pytest.importorskip("stl")
        bad_file = tmp_path / "bad.stl"
        bad_file.write_text("this is not an stl file at all!!!")
        with pytest.raises(ValueError, match="Failed to parse STL file"):
            parse_stl_file(str(bad_file))

    def test_parse_stl_exported_from_top_level(self):
        """parse_stl_file is accessible from the top-level package."""
        import pykarambola
        assert hasattr(pykarambola, "parse_stl_file")

    def test_is_stl_file(self):
        assert is_stl_file("test.stl")
        assert is_stl_file("test.STL")
        assert not is_stl_file("test.obj")
        assert not is_stl_file("test.off")
