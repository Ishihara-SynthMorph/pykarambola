"""
Port of test_box.cpp - tests Minkowski tensors on box geometries.
"""

import os
import math
import numpy as np
import pytest

from pykarambola.triangulation import LABEL_UNASSIGNED
from pykarambola.io_poly import parse_poly_file
from pykarambola.surface import check_surface
from pykarambola.results import CalcOptions
from pykarambola.minkowski import (
    calculate_w000, calculate_w100, calculate_w200, calculate_w300,
    calculate_w010, calculate_w110, calculate_w210, calculate_w310,
    calculate_w020, calculate_w120, calculate_w220, calculate_w320,
    calculate_w102, calculate_w202,
)
from pykarambola.eigensystem import calculate_eigensystem

TEST_INPUTS = os.path.join(os.path.dirname(__file__), "fixtures")


def _load_box(filename):
    """Load and prepare a box surface from a .poly file."""
    filepath = os.path.join(TEST_INPUTS, filename)
    surface = parse_poly_file(filepath, with_labels=False)
    surface.create_vertex_polygon_lookup_table()
    surface.create_polygon_polygon_lookup_table()
    co = CalcOptions()
    co.labels_set = False
    check_surface(co, surface)
    return surface


def _sorted_eigenvalues(w_matrix, label):
    """Get sorted eigenvalues for a matrix result."""
    eigsys = calculate_eigensystem(w_matrix)
    vals = sorted(eigsys[label].result.eigen_values)
    return np.array(vals)


class TestBox:
    """Test case: axis-aligned box with a=2, b=3, c=4."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.surface = _load_box("box_a=2_b=3_c=4.poly")
        self.a, self.b, self.c = 2.0, 3.0, 4.0
        self.pi = math.pi
        self.label = LABEL_UNASSIGNED

    def test_w000(self):
        """Volume = a*b*c."""
        w000 = calculate_w000(self.surface)
        assert w000[self.label].result == pytest.approx(
            self.a * self.b * self.c, rel=1e-4)

    def test_w100(self):
        """Surface area = 2/3 * (ab + ac + bc)."""
        w100 = calculate_w100(self.surface)
        expected = 2.0 / 3.0 * (self.a * self.b + self.a * self.c + self.b * self.c)
        assert w100[self.label].result == pytest.approx(expected, rel=1e-4)

    def test_w200(self):
        """Mean curvature = pi/3 * (a+b+c)."""
        w200 = calculate_w200(self.surface)
        expected = self.pi / 3.0 * (self.a + self.b + self.c)
        assert w200[self.label].result == pytest.approx(expected, rel=1e-4)

    def test_w300(self):
        """Euler characteristic = 4*pi/3."""
        w300 = calculate_w300(self.surface)
        expected = 4.0 * self.pi / 3.0
        assert w300[self.label].result == pytest.approx(expected, rel=1e-4)

    def test_w010(self):
        """Position vector = (0, 0, 0) for centered box."""
        w010 = calculate_w010(self.surface)
        np.testing.assert_allclose(w010[self.label].result, [0, 0, 0], atol=1e-3)

    def test_w110(self):
        """Surface-weighted position = (0, 0, 0)."""
        w110 = calculate_w110(self.surface)
        np.testing.assert_allclose(w110[self.label].result, [0, 0, 0], atol=1e-3)

    def test_w210(self):
        """Curvature-weighted position = (0, 0, 0)."""
        w210 = calculate_w210(self.surface)
        np.testing.assert_allclose(w210[self.label].result, [0, 0, 0], atol=1e-3)

    def test_w310(self):
        """Gaussian curvature-weighted position = (0, 0, 0)."""
        w310 = calculate_w310(self.surface)
        np.testing.assert_allclose(w310[self.label].result, [0, 0, 0], atol=1e-3)

    def test_w020_eigenvalues(self):
        """w020 eigenvalues = [a^3*b*c/12, b^3*a*c/12, c^3*a*b/12]."""
        a, b, c = self.a, self.b, self.c
        w020 = calculate_w020(self.surface)
        expected = sorted([a**3 * b * c / 12, b**3 * a * c / 12, c**3 * a * b / 12])
        actual = _sorted_eigenvalues(w020, self.label)
        np.testing.assert_allclose(actual, expected, rtol=1e-4)

    def test_w120_eigenvalues(self):
        """w120 eigenvalues."""
        a, b, c = self.a, self.b, self.c
        w120 = calculate_w120(self.surface)
        expected = sorted([
            1.0/6 * (1.0/3 * a**3 * (b + c) + a**2 * b * c),
            1.0/6 * (1.0/3 * b**3 * (a + c) + b**2 * a * c),
            1.0/6 * (1.0/3 * c**3 * (a + b) + c**2 * a * b),
        ])
        actual = _sorted_eigenvalues(w120, self.label)
        np.testing.assert_allclose(actual, expected, rtol=1e-4)

    def test_w220_eigenvalues(self):
        """w220 eigenvalues."""
        a, b, c = self.a, self.b, self.c
        pi = self.pi
        w220 = calculate_w220(self.surface)
        expected = sorted([
            pi / 36 * (a**3 + 3 * a**2 * (b + c)),
            pi / 36 * (b**3 + 3 * b**2 * (a + c)),
            pi / 36 * (c**3 + 3 * c**2 * (a + b)),
        ])
        actual = _sorted_eigenvalues(w220, self.label)
        np.testing.assert_allclose(actual, expected, rtol=1e-4)

    def test_w320_eigenvalues(self):
        """w320 eigenvalues = [pi/3*a^2, pi/3*b^2, pi/3*c^2]."""
        a, b, c = self.a, self.b, self.c
        pi = self.pi
        w320 = calculate_w320(self.surface)
        expected = sorted([pi / 3 * a**2, pi / 3 * b**2, pi / 3 * c**2])
        actual = _sorted_eigenvalues(w320, self.label)
        np.testing.assert_allclose(actual, expected, rtol=1e-4)

    def test_w102_eigenvalues(self):
        """w102 eigenvalues = [2ab/3, 2ac/3, 2bc/3]."""
        a, b, c = self.a, self.b, self.c
        w102 = calculate_w102(self.surface)
        expected = sorted([2.0/3 * a * b, 2.0/3 * a * c, 2.0/3 * b * c])
        actual = _sorted_eigenvalues(w102, self.label)
        np.testing.assert_allclose(actual, expected, rtol=1e-4)

    def test_w202_eigenvalues(self):
        """w202 eigenvalues = [pi/6*(a+b), pi/6*(a+c), pi/6*(b+c)]."""
        a, b, c = self.a, self.b, self.c
        pi = self.pi
        w202 = calculate_w202(self.surface)
        expected = sorted([pi / 6 * (a + b), pi / 6 * (a + c), pi / 6 * (b + c)])
        actual = _sorted_eigenvalues(w202, self.label)
        np.testing.assert_allclose(actual, expected, rtol=1e-4)


class TestSchiefeBox:
    """Test case: skewed box with a=2, b=3, c=4 (rotation invariance)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.surface = _load_box("schiefebox_a=2_b=3_c=4.poly")
        self.a, self.b, self.c = 2.0, 3.0, 4.0
        self.pi = math.pi
        self.label = LABEL_UNASSIGNED

    def test_w000(self):
        w000 = calculate_w000(self.surface)
        assert w000[self.label].result == pytest.approx(
            self.a * self.b * self.c, rel=1e-4)

    def test_w100(self):
        w100 = calculate_w100(self.surface)
        expected = 2.0 / 3.0 * (self.a * self.b + self.a * self.c + self.b * self.c)
        assert w100[self.label].result == pytest.approx(expected, rel=1e-4)

    def test_w200(self):
        w200 = calculate_w200(self.surface)
        expected = self.pi / 3.0 * (self.a + self.b + self.c)
        assert w200[self.label].result == pytest.approx(expected, rel=1e-4)

    def test_w300(self):
        w300 = calculate_w300(self.surface)
        expected = 4.0 * self.pi / 3.0
        assert w300[self.label].result == pytest.approx(expected, rel=1e-4)

    def test_w010(self):
        w010 = calculate_w010(self.surface)
        np.testing.assert_allclose(w010[self.label].result, [0, 0, 0], atol=1e-3)

    def test_w110(self):
        w110 = calculate_w110(self.surface)
        np.testing.assert_allclose(w110[self.label].result, [0, 0, 0], atol=1e-3)

    def test_w210(self):
        w210 = calculate_w210(self.surface)
        np.testing.assert_allclose(w210[self.label].result, [0, 0, 0], atol=1e-3)

    def test_w310(self):
        w310 = calculate_w310(self.surface)
        np.testing.assert_allclose(w310[self.label].result, [0, 0, 0], atol=1e-3)

    def test_w020_eigenvalues(self):
        a, b, c = self.a, self.b, self.c
        w020 = calculate_w020(self.surface)
        expected = sorted([a**3 * b * c / 12, b**3 * a * c / 12, c**3 * a * b / 12])
        actual = _sorted_eigenvalues(w020, self.label)
        np.testing.assert_allclose(actual, expected, rtol=5e-3)

    def test_w120_eigenvalues(self):
        a, b, c = self.a, self.b, self.c
        w120 = calculate_w120(self.surface)
        expected = sorted([
            1.0/6 * (1.0/3 * a**3 * (b + c) + a**2 * b * c),
            1.0/6 * (1.0/3 * b**3 * (a + c) + b**2 * a * c),
            1.0/6 * (1.0/3 * c**3 * (a + b) + c**2 * a * b),
        ])
        actual = _sorted_eigenvalues(w120, self.label)
        np.testing.assert_allclose(actual, expected, rtol=1e-4)

    def test_w220_eigenvalues(self):
        a, b, c = self.a, self.b, self.c
        pi = self.pi
        w220 = calculate_w220(self.surface)
        expected = sorted([
            pi / 36 * (a**3 + 3 * a**2 * (b + c)),
            pi / 36 * (b**3 + 3 * b**2 * (a + c)),
            pi / 36 * (c**3 + 3 * c**2 * (a + b)),
        ])
        actual = _sorted_eigenvalues(w220, self.label)
        np.testing.assert_allclose(actual, expected, rtol=1e-4)

    def test_w320_eigenvalues(self):
        a, b, c = self.a, self.b, self.c
        pi = self.pi
        w320 = calculate_w320(self.surface)
        expected = sorted([pi / 3 * a**2, pi / 3 * b**2, pi / 3 * c**2])
        actual = _sorted_eigenvalues(w320, self.label)
        np.testing.assert_allclose(actual, expected, rtol=1e-4)

    def test_w102_eigenvalues(self):
        a, b, c = self.a, self.b, self.c
        w102 = calculate_w102(self.surface)
        expected = sorted([2.0/3 * a * b, 2.0/3 * a * c, 2.0/3 * b * c])
        actual = _sorted_eigenvalues(w102, self.label)
        np.testing.assert_allclose(actual, expected, rtol=1e-4)

    def test_w202_eigenvalues(self):
        a, b, c = self.a, self.b, self.c
        pi = self.pi
        w202 = calculate_w202(self.surface)
        expected = sorted([pi / 6 * (a + b), pi / 6 * (a + c), pi / 6 * (b + c)])
        actual = _sorted_eigenvalues(w202, self.label)
        np.testing.assert_allclose(actual, expected, rtol=1e-4)
