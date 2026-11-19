# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2026 Keisuke Ishihara, Yajushi Khurana
# This file is part of pykarambola.
# See LICENSE for the full license text.
"""
Comprehensive tests for 2D Irreducible Minkowski Tensors (pykarambola.imt2d),
including Papaya2 golden-reference validation (polygons, 16 marching-squares
cases, potato.png, Minkowski maps, Voronoi point patterns) and 2D label images.
"""

import os
import warnings

import numpy as np
import pytest

import pykarambola as pk
from pykarambola import (
    imt_image,
    imt_label_image,
    imt_pointpattern,
    imt_polygon,
    minkowski_map,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_test_photo(pattern):
    """Parse Papaya2's ASCII TestPhoto format into Cartesian (x, y) array.

    Rows are separated by '_', ordered top-to-bottom (so last row is y=0),
    matching test_image.cpp's TestPhoto.
    """
    rows = [r for r in pattern.split("_") if r != ""]
    ny = len(rows)
    nx = len(rows[0])
    arr_xy = np.zeros((nx, ny), dtype=np.float64)
    for row_idx, row in enumerate(rows):
        y = ny - 1 - row_idx
        for x, ch in enumerate(row):
            arr_xy[x, y] = 1.0 if ch == "x" else 0.0
    return arr_xy


def _mock_image_xy():
    """Exact mock_image() from Papaya2's python/test/image_test.py (shape (3, 2))."""
    return np.transpose([
        [0.2, 0.4, 0.1],
        [0.7, 0.3, 0.6],
    ]).astype(np.float64)


# ---------------------------------------------------------------------------
# 1. Top-level API & imt_polygon
# ---------------------------------------------------------------------------

class TestImtPolygon:
    # Papaya2 concave 5-gon from test_imt.cpp and python/test/poly_test.py
    CONCAVE_VERTS = np.array([
        [2.0, 3.0],
        [0.25, 0.5],
        [-3.0, 4.0],
        [-2.0, 0.0],
        [2.0, 0.0],
    ])
    CONCAVE_AREA = 67.0 / 8.0
    CONCAVE_PSI = np.array([
        18.9509878231867863325879476083 + 0.0j,
        0.0 + 0.0j,
        4.03586622195516544754949115540 + 3.83613097404496685735871370308j,
        -2.25242470620683687651763840592 + 7.91410261263648605530287871331j,
        2.23638454508277526919840618339 + 2.16586100869600377036856029938j,
        0.70071569649771790984433739720 - 10.30885074737037023789259031490j,
        -4.28085548650291806830723860758 + 0.96683279082320198340559106474j,
    ])

    def test_top_level_exports(self):
        for name in (
            "imt_polygon",
            "imt_image",
            "imt_label_image",
            "minkowski_map",
            "imt_pointpattern",
        ):
            assert hasattr(pk, name)

    def test_concave_polygon_reference(self):
        res = imt_polygon(self.CONCAVE_VERTS)
        assert set(res.keys()) == {"area", "perimeter", "beta102", "isoper", "psi", "q"}
        assert res["psi"].shape == (13,)
        assert res["psi"].dtype == np.complex128
        assert res["q"].shape == (13,)
        assert res["q"].dtype == np.float64

        assert res["area"] == pytest.approx(self.CONCAVE_AREA, rel=1e-14, abs=1e-14)
        assert res["perimeter"] == pytest.approx(self.CONCAVE_PSI[0].real, rel=1e-14, abs=1e-14)
        assert res["psi"][0] == pytest.approx(self.CONCAVE_PSI[0], rel=1e-14, abs=1e-14)
        assert abs(res["psi"][1]) < 1e-14
        assert res["q"][0] == pytest.approx(1.0, abs=1e-15)
        for s in range(2, 7):
            assert res["psi"][s] == pytest.approx(self.CONCAVE_PSI[s], rel=1e-13, abs=1e-13)
            expected_q = abs(self.CONCAVE_PSI[s]) / self.CONCAVE_PSI[0].real
            assert res["q"][s] == pytest.approx(expected_q, rel=1e-14, abs=1e-14)

        q2 = abs(self.CONCAVE_PSI[2]) / self.CONCAVE_PSI[0].real
        assert res["beta102"] == pytest.approx((1.0 - q2) / (1.0 + q2), rel=1e-14)
        assert res["isoper"] == pytest.approx(
            4.0 * np.pi * self.CONCAVE_AREA / (self.CONCAVE_PSI[0].real ** 2), rel=1e-14
        )

    def test_concave_polygon_relabeling_and_zero_length_edges(self):
        n = len(self.CONCAVE_VERTS)
        for shift in range(n):
            relabelled = np.roll(self.CONCAVE_VERTS, shift, axis=0)
            res = imt_polygon(relabelled)
            assert res["area"] == pytest.approx(self.CONCAVE_AREA, abs=1e-14)
            assert np.allclose(res["psi"][:7], self.CONCAVE_PSI, atol=1e-13, rtol=1e-13)

            # Insert a duplicate vertex (zero-length edge) at position `shift`
            with_dup = np.insert(self.CONCAVE_VERTS, shift, self.CONCAVE_VERTS[shift], axis=0)
            res_dup = imt_polygon(with_dup)
            assert res_dup["area"] == pytest.approx(self.CONCAVE_AREA, abs=1e-14)
            assert np.allclose(res_dup["psi"][:7], self.CONCAVE_PSI, atol=1e-13, rtol=1e-13)

    def test_equilateral_triangle_analytic(self):
        verts = np.array([
            [1.0, 0.0],
            [np.cos(2.0 * np.pi / 3.0), np.sin(2.0 * np.pi / 3.0)],
            [np.cos(-2.0 * np.pi / 3.0), np.sin(-2.0 * np.pi / 3.0)],
        ])
        res = imt_polygon(verts)
        side = np.sqrt(3.0)
        assert res["area"] == pytest.approx(3.0 * np.sqrt(3.0) / 4.0, rel=1e-14)
        assert res["perimeter"] == pytest.approx(3.0 * side, rel=1e-14)
        for s in range(2, 13):
            ref = side * (
                np.exp(1j * (np.pi / 3.0) * s)
                + np.exp(-1j * (np.pi / 3.0) * s)
                + np.exp(1j * np.pi * s)
            )
            assert res["psi"][s] == pytest.approx(ref, abs=1e-13)
        # 3-fold symmetry: q_3 = q_6 = q_9 = q_12 = 1, others 0
        for s in range(2, 13):
            expected = 1.0 if s % 3 == 0 else 0.0
            assert res["q"][s] == pytest.approx(expected, abs=1e-14)

    def test_rectangle_and_square_analytic(self):
        a, b = 3.0, 7.0
        rect = np.array([[0.0, 0.0], [a, 0.0], [a, b], [0.0, b]])
        res = imt_polygon(rect)
        assert res["area"] == pytest.approx(a * b, rel=1e-15)
        assert res["perimeter"] == pytest.approx(2.0 * (a + b), rel=1e-15)
        assert res["beta102"] == pytest.approx(min(a, b) / max(a, b), rel=1e-14)
        assert res["isoper"] == pytest.approx(4.0 * np.pi * a * b / (2.0 * (a + b)) ** 2, rel=1e-14)

        sq = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
        res_sq = imt_polygon(sq)
        assert res_sq["beta102"] == pytest.approx(1.0, abs=1e-15)
        assert res_sq["isoper"] == pytest.approx(np.pi / 4.0, rel=1e-15)
        assert res_sq["q"][4] == pytest.approx(1.0, abs=1e-15)
        assert res_sq["q"][2] == pytest.approx(0.0, abs=1e-15)

    def test_ellipse_3_8_and_circle(self):
        n = 30000
        phi = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        # Unit circle
        circ = np.column_stack([np.cos(phi), np.sin(phi)])
        res_c = imt_polygon(circ, max_s=6)
        assert res_c["area"] == pytest.approx(np.pi, rel=1e-6)
        assert res_c["isoper"] == pytest.approx(1.0, rel=1e-6)
        assert np.all(res_c["q"][2:7] < 1e-8)

        # 3:8 ellipse from test_imt.cpp
        e = np.sqrt(1.0 - 9.0 / 64.0)
        r = (1.0 - (e * np.cos(phi)) ** 2) ** (-0.5)
        ell = np.column_stack([r * np.cos(phi), r * np.sin(phi)])
        res_e = imt_polygon(ell, max_s=8)
        p = res_e["perimeter"]
        assert res_e["psi"][2] / p == pytest.approx(-0.6307002615184807, abs=1e-6)
        assert res_e["psi"][3] / p == pytest.approx(0.0, abs=1e-6)
        assert res_e["psi"][4] / p == pytest.approx(0.3484450247890247, abs=1e-6)
        assert res_e["psi"][6] / p == pytest.approx(-0.1821171064194439, abs=1e-6)
        assert res_e["psi"][8] / p == pytest.approx(0.0923027297442122, abs=1e-6)

    def test_covariance_under_translation_rotation_scaling(self):
        v = self.CONCAVE_VERTS
        base = imt_polygon(v)

        # Far translation
        shifted = imt_polygon(v + np.array([1e7, -2e7]))
        assert shifted["area"] == pytest.approx(base["area"], rel=1e-12)
        assert np.allclose(shifted["psi"], base["psi"], atol=1e-11, rtol=1e-11)

        # Rotation by theta multiplies psi[s] by exp(i * s * theta)
        theta = 0.73
        c, s_th = np.cos(theta), np.sin(theta)
        rot_mat = np.array([[c, -s_th], [s_th, c]])
        rotated = imt_polygon(v @ rot_mat.T)
        assert rotated["area"] == pytest.approx(base["area"], rel=1e-13)
        assert rotated["perimeter"] == pytest.approx(base["perimeter"], rel=1e-13)
        assert np.allclose(rotated["q"], base["q"], atol=1e-13)
        orders = np.arange(13)
        assert np.allclose(rotated["psi"], base["psi"] * np.exp(1j * orders * theta), atol=1e-12)

        # Uniform scaling by lam
        lam = 3.5
        scaled = imt_polygon(v * lam)
        assert scaled["area"] == pytest.approx(base["area"] * lam**2, rel=1e-13)
        assert scaled["perimeter"] == pytest.approx(base["perimeter"] * lam, rel=1e-13)
        assert np.allclose(scaled["psi"], base["psi"] * lam, atol=1e-12)
        assert np.allclose(scaled["q"], base["q"], atol=1e-13)
        assert scaled["beta102"] == pytest.approx(base["beta102"], rel=1e-13)
        assert scaled["isoper"] == pytest.approx(base["isoper"], rel=1e-13)

    def test_cw_and_ensure_ccw(self):
        ccw = imt_polygon(self.CONCAVE_VERTS)
        cw_raw = imt_polygon(self.CONCAVE_VERTS[::-1], ensure_ccw=False)
        cw_fixed = imt_polygon(self.CONCAVE_VERTS[::-1], ensure_ccw=True)

        assert cw_raw["area"] == pytest.approx(-ccw["area"], rel=1e-14)
        assert cw_fixed["area"] == pytest.approx(ccw["area"], rel=1e-14)
        assert np.allclose(cw_fixed["psi"], ccw["psi"], atol=1e-13)

    def test_max_s_and_degenerate_polygon(self):
        res0 = imt_polygon(self.CONCAVE_VERTS, max_s=0)
        assert res0["psi"].shape == (1,)
        assert res0["q"].shape == (1,)
        assert np.isnan(res0["beta102"])

        res20 = imt_polygon(self.CONCAVE_VERTS, max_s=20)
        assert res20["psi"].shape == (21,)
        assert res20["q"].shape == (21,)

        # Coincident vertices -> perimeter = 0 -> NaN for q, beta102, isoper
        deg = imt_polygon([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        assert deg["area"] == 0.0
        assert deg["perimeter"] == 0.0
        assert np.all(np.isnan(deg["q"]))
        assert np.isnan(deg["beta102"])
        assert np.isnan(deg["isoper"])

    def test_validation_errors(self):
        with pytest.raises(ValueError):
            imt_polygon([[]])
        with pytest.raises(ValueError):
            imt_polygon([[0.0, 0.0]])
        with pytest.raises(ValueError):
            imt_polygon([[0.0], [1.0]])
        with pytest.raises(ValueError):
            imt_polygon([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
        with pytest.raises(ValueError):
            imt_polygon([0.0, 1.0])
        with pytest.raises(ValueError):
            imt_polygon(self.CONCAVE_VERTS, max_s=-1)
        with pytest.raises(ValueError):
            imt_polygon(self.CONCAVE_VERTS, max_s=True)
        with pytest.raises(ValueError, match="finite"):
            imt_polygon([[0.0, 0.0], [np.nan, 0.0], [0.0, 1.0]])
        with pytest.raises(ValueError, match="finite"):
            imt_polygon([[0.0, 0.0], [np.inf, 0.0], [0.0, 1.0]])


# ---------------------------------------------------------------------------
# 2. imt_image (marching squares, flags, boundaries, potato.png)
# ---------------------------------------------------------------------------

class TestImtImage:
    CONFIGS = (
        "  _  _", "  _x _", "x _  _", "x _x _",
        "  _ x_", "  _xx_", "x _ x_", "x _xx_",
        " x_  _", " x_x _", "xx_  _", "xx_x _",
        " x_ x_", " x_xx_", "xx_ x_", "xx_xx_",
    )
    REF_VOLUMES = (
        0.0, 0.125, 0.125, 0.5, 0.125, 0.5,
        0.75, 0.875, 0.125, 0.75, 0.5, 0.875,
        0.5, 0.875, 0.875, 1.0,
    )

    def test_all_16_neighborhoods_and_complementarity(self):
        dx, dy = 2.0, 2.0
        pix_area = dx * dy
        for cfg, ref_vol in zip(self.CONFIGS, self.REF_VOLUMES):
            ph = _parse_test_photo(cfg)
            assert ph.shape == (2, 2)

            # Regular at 0.75 and interpolated at 0.5
            r_imt = imt_image(ph, 0.75, spacing=(dx, dy), boundary=None, method="regular", xy_order=True)
            i_imt = imt_image(ph, 0.5, spacing=(dx, dy), boundary=None, method="interpolated", xy_order=True)
            assert r_imt["area"] == pytest.approx(ref_vol * pix_area, abs=1e-14)
            assert i_imt["area"] == pytest.approx(ref_vol * pix_area, abs=1e-14)

            # Complementarity at threshold 1.0 for both methods and saddle policies
            for method in ("regular", "interpolated"):
                for connect in ("white", "black"):
                    w = imt_image(
                        ph, 1.0, spacing=(dx, dy), boundary=None,
                        method=method, analyze="white", connect=connect, xy_order=True,
                    )
                    b = imt_image(
                        ph, 1.0, spacing=(dx, dy), boundary=None,
                        method=method, analyze="black", connect=connect, xy_order=True,
                    )
                    assert w["area"] + b["area"] == pytest.approx(pix_area, abs=1e-14)
                    assert b["psi"][0] == pytest.approx(+w["psi"][0], abs=1e-14)
                    assert b["psi"][2] == pytest.approx(+w["psi"][2], abs=1e-14)
                    assert b["psi"][3] == pytest.approx(-w["psi"][3], abs=1e-14)

    def test_grayscale_saddle_and_normal_flip(self):
        # Exact cases from test_image.cpp lines 241-307 (configs #12, #6, #9)
        cases = [
            np.array([[-2.1, -1.3], [1.7, 0.7]]),   # config #12
            np.array([[-2.1, 0.7], [1.7, -1.3]]),   # critical config #6
            np.array([[2.1, -0.7], [-1.7, 1.3]]),   # critical config #9
        ]
        for arr in cases:
            for connect in ("white", "black"):
                w = imt_image(arr, 0.0, boundary=None, analyze="white", connect=connect, xy_order=True)
                b = imt_image(arr, 0.0, boundary=None, analyze="black", connect=connect, xy_order=True)
                assert w["area"] + b["area"] == pytest.approx(1.0, abs=1e-14)
                assert w["perimeter"] == pytest.approx(b["perimeter"], abs=1e-14)
                assert w["psi"][2] == pytest.approx(+b["psi"][2], abs=1e-14)
                assert w["psi"][3] == pytest.approx(-b["psi"][3], abs=1e-14)

        # Saddle connect='white' vs connect='black' must differ on configs #6 and #9
        for arr in cases[1:]:
            cw = imt_image(arr, 0.0, boundary=None, connect="white", xy_order=True)
            cb = imt_image(arr, 0.0, boundary=None, connect="black", xy_order=True)
            assert cw["area"] > cb["area"]

    def test_single_pixel_and_exact_threshold_corner_case(self):
        tp_side = 2.0
        tp_area = tp_side ** 2
        one_pix = _parse_test_photo("   _ x _   _")

        for method, thr in [("interpolated", 0.5), ("regular", 0.75)]:
            res = imt_image(one_pix, thr, spacing=(tp_side, tp_side), boundary=None, method=method, xy_order=True)
            assert res["area"] == pytest.approx(0.5 * tp_area, abs=1e-14)
            assert res["perimeter"] == pytest.approx(2.0 * np.sqrt(2.0) * tp_side, abs=1e-14)
            assert res["q"][2] == pytest.approx(0.0, abs=1e-14)
            assert res["q"][3] == pytest.approx(0.0, abs=1e-14)
            assert res["q"][4] == pytest.approx(1.0, abs=1e-14)
            assert res["psi"][4] == pytest.approx(-2.0 * np.sqrt(2.0) * tp_side, abs=1e-14)

        # Threshold matches pixel value 1.0 exactly
        res_eq_w = imt_image(one_pix, 1.0, spacing=(tp_side, tp_side), boundary=None, xy_order=True)
        assert res_eq_w["area"] == 0.0
        assert res_eq_w["perimeter"] == 0.0
        assert np.all(res_eq_w["psi"] == 0.0)

        res_eq_b = imt_image(
            one_pix, 1.0, spacing=(tp_side, tp_side), boundary=None, analyze="black", xy_order=True
        )
        assert res_eq_b["area"] == pytest.approx(4.0 * tp_area, abs=1e-14)
        assert res_eq_b["perimeter"] == 0.0

    def test_tetromino_orientations_and_padding(self):
        tp_side = 2.0
        tp_area = tp_side ** 2
        ref_area = 3.5 * tp_area
        ref_peri = (4.0 * np.sqrt(2.0) + 2.0) * tp_side

        # Orientation #1
        ph1 = _parse_test_photo("    _ x  _ xx _ x  _    _")
        res1 = imt_image(ph1, 0.5, spacing=(tp_side, tp_side), boundary=None, xy_order=True)
        assert res1["area"] == pytest.approx(ref_area, abs=1e-13)
        assert res1["perimeter"] == pytest.approx(ref_peri, abs=1e-13)
        assert res1["psi"][2] == pytest.approx(2.0 * tp_side, abs=1e-13)
        assert res1["psi"][3] == pytest.approx(-4.0 * tp_side, abs=1e-13)
        assert res1["psi"][4] == pytest.approx((2.0 - 4.0 * np.sqrt(2.0)) * tp_side, abs=1e-13)

        # Orientation #2 with padding (touching boundary)
        ph2_touch = _parse_test_photo("    _ x  _xxx _")
        res2 = imt_image(ph2_touch, 0.5, spacing=(tp_side, tp_side), boundary=0.0, xy_order=True)
        assert res2["area"] == pytest.approx(ref_area, abs=1e-13)
        assert res2["perimeter"] == pytest.approx(ref_peri, abs=1e-13)
        assert res2["psi"][2] == pytest.approx(-2.0 * tp_side, abs=1e-13)
        assert res2["psi"][3] == pytest.approx(4.0j * tp_side, abs=1e-13)
        assert res2["psi"][4] == pytest.approx((2.0 - 4.0 * np.sqrt(2.0)) * tp_side, abs=1e-13)

    def test_potato_png_golden(self, test_inputs_dir):
        PIL_Image = pytest.importorskip("PIL.Image")
        png_path = os.path.join(test_inputs_dir, "potato.png")
        raw = np.asarray(PIL_Image.open(png_path))
        if raw.ndim == 3:
            raw = raw[:, :, 0]
        # Papaya2 Cartesian layout: shape (302, 361)
        img_xy = raw.T[:, ::-1].astype(np.float64)
        assert img_xy.shape == (302, 361)

        thr = 1.469734492275599e2
        res_xy = imt_image(img_xy, threshold=thr, boundary=0.0, xy_order=True)

        ref_area = 7.760018008530601e4
        ref_peri = 1.009820521813200e3
        ref_q_arg = {
            2: (1.563708314579508e-1, -5.574818755549115e-1),
            3: (3.980635759575531e-2, 4.577327494757427e-1),
            4: (1.481665040254243e-1, 7.141179567742937e-1),
            5: (1.922804813953316e-1, 2.070654574313143e0),
            6: (1.454253098691293e-1, -2.519436436486195e-1),
            7: (7.671564980492507e-3, 1.404644402590062e0),
            8: (1.860509319491248e-1, 7.356981574804343e-2),
        }

        assert res_xy["area"] == pytest.approx(ref_area, rel=1e-13)
        assert res_xy["perimeter"] == pytest.approx(ref_peri, rel=1e-13)
        for s, (q_ref, arg_ref) in ref_q_arg.items():
            assert res_xy["q"][s] == pytest.approx(q_ref, rel=1e-12, abs=1e-13)
            assert np.angle(res_xy["psi"][s]) == pytest.approx(arg_ref, abs=1e-12)

        # Default bioimage (row, col) on unflipped raw PIL array:
        # all rotation invariants identical, psi is exact complex conjugate
        res_rc = imt_image(raw.astype(np.float64), threshold=thr, boundary=0.0, xy_order=False)
        assert res_rc["area"] == pytest.approx(ref_area, rel=1e-13)
        assert res_rc["perimeter"] == pytest.approx(ref_peri, rel=1e-13)
        assert np.allclose(res_rc["q"], res_xy["q"], rtol=1e-13, atol=1e-14)
        assert np.allclose(np.conj(res_rc["psi"]), res_xy["psi"], rtol=1e-13, atol=1e-12)

        # Normalized [0, 1] default threshold = 0.5 (from image_test.py)
        res_norm = imt_image(img_xy / 255.0, xy_order=True)
        assert res_norm["area"] == pytest.approx(7.7696018577601324e4, rel=1e-13)
        assert res_norm["perimeter"] == pytest.approx(1.0092019379255355e3, rel=1e-13)
        assert res_norm["psi"][2] == pytest.approx(
            1.3389577638821120e2 - 8.4186446356513883e1j, rel=1e-12
        )

        # Multi-threshold vector call
        res_multi = imt_image(img_xy, threshold=[thr, 127.5], xy_order=True)
        assert res_multi["area"].shape == (2,)
        assert res_multi["psi"].shape == (2, 13)
        assert res_multi["q"].shape == (2, 13)
        assert res_multi["area"][0] == pytest.approx(ref_area, rel=1e-13)
        assert res_multi["area"][1] == pytest.approx(7.7696018577601324e4, rel=1e-13)

    def test_periodic_boundary_and_shift_invariance(self):
        rng = np.random.default_rng(7)
        img = rng.normal(size=(24, 30))
        spacing = (1.5, 0.75)

        # All-ones periodic image has exact area = H * W * sy * sx, zero perimeter
        ones = np.ones((10, 12))
        res_ones = imt_image(ones, threshold=0.5, spacing=spacing, boundary="periodic")
        assert res_ones["area"] == pytest.approx(10 * 12 * 1.5 * 0.75, rel=1e-14)
        assert res_ones["perimeter"] == 0.0

        # Periodic IMT is invariant under arbitrary cyclic rolls
        base = imt_image(img, threshold=0.1, spacing=spacing, boundary="periodic")
        rolled = imt_image(
            np.roll(img, shift=(7, -11), axis=(0, 1)),
            threshold=0.1,
            spacing=spacing,
            boundary="periodic",
        )
        assert rolled["area"] == pytest.approx(base["area"], rel=1e-13)
        assert rolled["perimeter"] == pytest.approx(base["perimeter"], rel=1e-13)
        assert np.allclose(rolled["psi"], base["psi"], atol=1e-12, rtol=1e-12)

    def test_xy_order_equivalence_and_anisotropic_spacing(self):
        rng = np.random.default_rng(99)
        img_rc = rng.uniform(size=(20, 35))
        sy, sx = 2.5, 0.8

        res_rc = imt_image(img_rc, threshold=0.4, spacing=(sy, sx), boundary=0.0, xy_order=False)
        res_xy = imt_image(img_rc.T, threshold=0.4, spacing=(sx, sy), boundary=0.0, xy_order=True)
        assert res_rc["area"] == pytest.approx(res_xy["area"], rel=1e-15)
        assert res_rc["perimeter"] == pytest.approx(res_xy["perimeter"], rel=1e-15)
        assert np.allclose(res_rc["psi"], res_xy["psi"], atol=1e-14, rtol=1e-14)

        # Stretching a binary square by (sy=1, sx=3) makes it 3x wider horizontally -> psi_2 < 0
        sq = np.zeros((20, 20))
        sq[5:15, 5:15] = 1.0
        wide = imt_image(sq, threshold=0.5, spacing=(1.0, 3.0), boundary=0.0)
        tall = imt_image(sq, threshold=0.5, spacing=(3.0, 1.0), boundary=0.0)
        assert wide["area"] == pytest.approx(tall["area"], rel=1e-14)
        assert wide["psi"][2].real < 0.0
        assert tall["psi"][2].real > 0.0
        assert wide["beta102"] == pytest.approx(tall["beta102"], rel=1e-14)

    def test_validation_errors(self):
        img = np.zeros((4, 4))
        with pytest.raises(ValueError):
            imt_image([0.0, 1.0])
        with pytest.raises(ValueError):
            imt_image(np.zeros((2, 2, 2)))
        with pytest.raises(ValueError):
            imt_image(img, threshold=np.eye(2))
        with pytest.raises(ValueError):
            imt_image(img, threshold="bad")
        with pytest.raises(ValueError):
            imt_image(img, boundary="invalid")
        with pytest.raises(ValueError):
            imt_image(img, boundary=[1.0, 2.0])
        with pytest.raises(ValueError):
            imt_image(img, method="unknown")
        with pytest.raises(ValueError):
            imt_image(img, analyze="gray")
        with pytest.raises(ValueError):
            imt_image(img, connect="both")
        with pytest.raises(ValueError):
            imt_image(img, spacing=(1.0, -1.0))
        with pytest.raises(ValueError):
            imt_image(img, spacing=(1.0, 2.0, 3.0))
        with pytest.raises(ValueError, match="finite"):
            imt_image(img, threshold=np.nan)
        with pytest.raises(ValueError, match="finite"):
            imt_image(img, threshold=[0.2, np.inf])
        with pytest.raises(ValueError, match="finite"):
            imt_image(img, boundary=np.nan)
        with pytest.raises(ValueError, match="finite"):
            imt_image(img, boundary=np.inf)

    def test_empty_multi_threshold(self):
        img = np.zeros((4, 5))
        res = imt_image(img, threshold=[], max_s=6)
        assert res["area"].shape == (0,)
        assert res["perimeter"].shape == (0,)
        assert res["beta102"].shape == (0,)
        assert res["isoper"].shape == (0,)
        assert res["psi"].shape == (0, 7)
        assert res["q"].shape == (0, 7)


# ---------------------------------------------------------------------------
# 3. minkowski_map
# ---------------------------------------------------------------------------

class TestMinkowskiMap:
    def test_periodic_stripe_reference_orders(self):
        # Exact test from Papaya2 python/test/image_test.py
        image = np.zeros((2, 3), dtype=np.float64)
        image[:, 1] = 1.0

        m0 = minkowski_map(image, boundary="periodic", s=0, xy_order=True)
        assert m0.shape == (2, 3)
        assert np.allclose(m0, [[1, 1, 0], [1, 1, 0]], atol=1e-14)

        m2 = minkowski_map(image, boundary="periodic", s=2, xy_order=True)
        assert np.allclose(m2, [[-1, -1, 0], [-1, -1, 0]], atol=1e-14)

        m3 = minkowski_map(image, boundary="periodic", s=3, xy_order=True)
        assert np.allclose(m3, [[1j, -1j, 0], [1j, -1j, 0]], atol=1e-14)

        m12 = minkowski_map(image, boundary="periodic", s=12, xy_order=True)
        assert np.allclose(m12, [[1, 1, 0], [1, 1, 0]], atol=1e-14)

    def test_mock_image_padded_and_periodic_reference(self):
        img = _mock_image_xy()
        assert img.shape == (3, 2)

        # Padded zeros (boundary=0.0)
        m_zero = minkowski_map(img, threshold=0.06, s=2, boundary=0.0, xy_order=True)
        assert m_zero.shape == (4, 3)
        expected_zero = np.array([
            [0.0 + 0.98994949j, 9.32903324e-1 + 0.41905818j, -2.87102621e-16 - 1.29299526j],
            [-9.66685285e-1 + 0.29668091j, 0.0 + 0.0j, -9.80555919e-1 + 0.22709318j],
            [-7.27257400e-1 - 0.82072935j, 0.0 + 0.0j, -9.85086818e-1 - 0.19900744j],
            [-3.45420341e-16 - 0.56568542j, 6.70820393e-1 - 0.89442719j, 0.0 + 1.27279221j],
        ])
        assert np.allclose(m_zero, expected_zero, atol=1e-6)

        # Custom padding value (boundary=0.03)
        m_pad = minkowski_map(img, threshold=0.06, s=2, boundary=0.03, xy_order=True)
        expected_pad = np.array([
            [0.0 + 1.16464646j, 9.74244511e-1 + 0.26113419j, 2.99957962e-16 - 1.35089057j],
            [-9.86423194e-1 + 0.18991693j, 0.0 + 0.0j, -9.93416388e-1 + 0.13237905j],
            [-8.30535726e-1 - 0.6564754j, 0.0 + 0.0j, -9.94880423e-1 - 0.11675958j],
            [-1.79439138e-16 - 0.80812204j, 8.03748429e-1 - 0.7037892j, 0.0 + 1.33978127j],
        ])
        assert np.allclose(m_pad, expected_pad, atol=1e-6)

        # Periodic multi-threshold
        m_per = minkowski_map(img, threshold=[0.6, 1.1], s=2, boundary="periodic", xy_order=True)
        assert m_per.shape == (2, 3, 2)
        expected_per = np.array([
            [-0.07027819 - 0.31234752j, -0.07027819 + 0.31234752j],
            [0.0 + 0.0j, 0.0 + 0.0j],
            [-0.94135745 + 0.39223227j, -0.94135745 - 0.39223227j],
        ])
        assert np.allclose(m_per[0], expected_per, atol=1e-6)
        assert np.allclose(m_per[1], 0.0, atol=1e-14)

    def test_map_sum_equals_global_psi(self):
        rng = np.random.default_rng(11)
        img = rng.normal(size=(25, 19))
        for boundary in (None, 0.0, -0.4, "periodic"):
            for method in ("interpolated", "regular"):
                for analyze in ("white", "black"):
                    for connect in ("white", "black"):
                        res = imt_image(
                            img, threshold=0.15, spacing=(1.3, 0.7), boundary=boundary,
                            method=method, analyze=analyze, connect=connect,
                        )
                        for s in (0, 1, 2, 3, 4, 12):
                            mm = minkowski_map(
                                img, threshold=0.15, s=s, spacing=(1.3, 0.7), boundary=boundary,
                                method=method, analyze=analyze, connect=connect,
                            )
                            assert np.sum(mm) == pytest.approx(res["psi"][s], rel=1e-12, abs=1e-12)

    def test_single_pixel_unpadded_empty_map_and_xy_transpose(self):
        one = np.array([[1.0]])
        mm_unpadded = minkowski_map(one, boundary=None)
        assert mm_unpadded.shape == (0, 0)
        mm_padded = minkowski_map(one, boundary=0.0)
        assert mm_padded.shape == (2, 2)
        assert np.all(mm_padded != 0.0)

        img_rc = _mock_image_xy().T
        mm_rc = minkowski_map(img_rc, threshold=0.06, boundary=0.0, xy_order=False)
        mm_xy = minkowski_map(img_rc.T, threshold=0.06, boundary=0.0, xy_order=True)
        assert mm_rc.shape == (3, 4)
        assert np.allclose(mm_rc, mm_xy.T, atol=1e-15)

    def test_validation_errors(self):
        img = _mock_image_xy()
        with pytest.raises(ValueError):
            minkowski_map(img, s=-1)
        with pytest.raises(ValueError):
            minkowski_map(img, s=0.5)
        with pytest.raises(ValueError):
            minkowski_map(img, s=[0, 2])
        with pytest.raises(ValueError):
            minkowski_map(img, s=True)
        with pytest.raises(ValueError):
            minkowski_map(img, boundary="nonsense")
        with pytest.raises(ValueError, match="finite"):
            minkowski_map(img, threshold=np.nan)
        with pytest.raises(ValueError, match="finite"):
            minkowski_map(img, threshold=[0.1, np.inf])
        with pytest.raises(ValueError, match="finite"):
            minkowski_map(img, boundary=np.nan)

    def test_empty_multi_threshold(self):
        img = np.zeros((4, 5))
        assert minkowski_map(img, threshold=[], boundary=None).shape == (0, 3, 4)
        assert minkowski_map(img, threshold=[], boundary=0.0).shape == (0, 5, 6)
        assert minkowski_map(img, threshold=[], boundary="periodic").shape == (0, 4, 5)


# ---------------------------------------------------------------------------
# 4. imt_label_image
# ---------------------------------------------------------------------------

class TestImtLabelImage:
    def _make_multi_label_image(self):
        lbl = np.zeros((64, 80), dtype=np.int32)
        # Label 5: disk in interior
        yy, xx = np.ogrid[:64, :80]
        lbl[(yy - 20) ** 2 + (xx - 25) ** 2 <= 10 ** 2] = 5
        # Label 42: rectangle touching top-left corner
        lbl[0:12, 0:18] = 42
        # Label 1000: ring (object with hole) touching bottom-right corner
        ring = ((yy - 55) ** 2 + (xx - 70) ** 2 <= 12 ** 2) & (
            (yy - 55) ** 2 + (xx - 70) ** 2 >= 5 ** 2
        )
        lbl[ring] = 1000
        return lbl

    def test_bbox_crop_exact_equivalence_to_full_image(self):
        lbl = self._make_multi_label_image()
        for pad in (True, False):
            for xy_order in (False, True):
                for method in ("interpolated", "regular"):
                    spacing = (1.5, 0.9)
                    res = imt_label_image(
                        lbl, spacing=spacing, pad=pad, method=method, xy_order=xy_order
                    )
                    assert set(res.keys()) == {5, 42, 1000}
                    for lab in (5, 42, 1000):
                        full = imt_image(
                            (lbl == lab).astype(np.float64),
                            threshold=0.5,
                            spacing=spacing,
                            boundary=0.0 if pad else None,
                            method=method,
                            xy_order=xy_order,
                        )
                        assert res[lab]["area"] == pytest.approx(full["area"], rel=1e-14, abs=1e-14)
                        assert res[lab]["perimeter"] == pytest.approx(
                            full["perimeter"], rel=1e-14, abs=1e-14
                        )
                        assert np.allclose(res[lab]["psi"], full["psi"], atol=1e-13, rtol=1e-13)
                        assert np.allclose(res[lab]["q"], full["q"], atol=1e-14, rtol=1e-14)

    def test_autolabel_connectivity(self):
        # Two diagonally touching squares
        img = np.zeros((12, 12), dtype=np.int32)
        img[2:5, 2:5] = 7
        img[5:8, 5:8] = 7

        # Default autolabel=False: single label 7
        res_single = imt_label_image(img, autolabel=False)
        assert list(res_single.keys()) == [7]

        # autolabel=True, connectivity=1 (4-connected): 2 separate objects
        res_c1 = imt_label_image(img, autolabel=True, connectivity=1)
        assert sorted(res_c1.keys()) == [1, 2]
        assert res_c1[1]["area"] == pytest.approx(res_c1[2]["area"], rel=1e-14)

        # autolabel=True, connectivity=2 (8-connected): 1 merged object
        res_c2 = imt_label_image(img, autolabel=True, connectivity=2)
        assert list(res_c2.keys()) == [1]

    def test_bool_mask_no_warning_and_autolabel_equivalence(self):
        # Standard boolean segmentation mask (e.g. elastin mask) must not warn
        lbl = self._make_multi_label_image()
        mask = lbl > 0

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res_single = imt_label_image(mask, autolabel=False)
            res_c1 = imt_label_image(mask, autolabel=True, connectivity=1)
            res_c2 = imt_label_image(mask, autolabel=True, connectivity=2)
        assert caught == []
        assert list(res_single.keys()) == [1]
        assert sorted(res_c1.keys()) == [1, 2, 3]
        assert sorted(res_c2.keys()) == [1, 2, 3]

        # Verify fast find_objects path (autolabel=True) matches full-image imt_image
        for pad in (True, False):
            for xy_order in (False, True):
                spacing = (1.4, 0.8)
                res = imt_label_image(
                    mask,
                    spacing=spacing,
                    pad=pad,
                    xy_order=xy_order,
                    autolabel=True,
                    connectivity=1,
                )
                assert sorted(res.keys()) == [1, 2, 3]
                # Sum of per-component areas and psi equals whole-mask imt_image
                # (components are pairwise disjoint and >1 pixel apart)
                whole = imt_image(
                    mask.astype(np.float64),
                    threshold=0.5,
                    spacing=spacing,
                    boundary=0.0 if pad else None,
                    xy_order=xy_order,
                )
                total_area = sum(r["area"] for r in res.values())
                total_peri = sum(r["perimeter"] for r in res.values())
                total_psi = sum(r["psi"] for r in res.values())
                assert total_area == pytest.approx(whole["area"], rel=1e-13, abs=1e-13)
                assert total_peri == pytest.approx(whole["perimeter"], rel=1e-13, abs=1e-13)
                assert np.allclose(total_psi, whole["psi"], atol=1e-12, rtol=1e-12)

    def test_empty_and_non_integer_warning(self):
        assert imt_label_image(np.zeros((10, 10), dtype=np.int32)) == {}
        assert imt_label_image(np.zeros((10, 10), dtype=bool), autolabel=True) == {}

        flt = np.zeros((10, 10), dtype=np.float64)
        flt[3:7, 3:7] = 1.0
        with pytest.warns(UserWarning, match="converting to int"):
            res = imt_label_image(flt)
        assert list(res.keys()) == [1]

    def test_validation_errors(self):
        with pytest.raises(ValueError):
            imt_label_image(np.zeros((4, 4, 4), dtype=np.int32))
        with pytest.raises(ValueError):
            imt_label_image(np.ones((4, 4), dtype=np.int32), autolabel=True, connectivity=3)
        with pytest.raises(ValueError, match="finite"):
            imt_label_image(np.ones((4, 4), dtype=np.int32), level=np.nan)
        with pytest.raises(ValueError, match="finite"):
            imt_label_image(np.ones((4, 4), dtype=np.int32), level=np.inf)


# ---------------------------------------------------------------------------
# 5. imt_pointpattern (Voronoi cells, periodic & non-periodic)
# ---------------------------------------------------------------------------

class TestImtPointPattern:
    def test_periodic_square_lattice(self):
        xs, ys = np.meshgrid(np.arange(4) + 0.5, np.arange(4) + 0.5)
        seeds = np.column_stack([xs.ravel(), ys.ravel()])
        res = imt_pointpattern(seeds, box=4.0)

        assert res["area"].shape == (16,)
        assert res["psi"].shape == (16, 13)
        assert res["q"].shape == (16, 13)
        assert np.allclose(res["area"], 1.0, atol=1e-13)
        assert np.allclose(res["perimeter"], 4.0, atol=1e-13)
        assert np.allclose(res["beta102"], 1.0, atol=1e-13)
        assert np.allclose(res["isoper"], np.pi / 4.0, atol=1e-13)
        assert np.allclose(res["q"][:, 4], 1.0, atol=1e-13)
        assert np.allclose(res["q"][:, 2], 0.0, atol=1e-13)

    def test_periodic_rectangular_and_single_seed(self):
        res_1 = imt_pointpattern([[0.3, 0.7]], box=[2.0, 5.0])
        assert res_1["area"][0] == pytest.approx(10.0, rel=1e-14)
        assert res_1["perimeter"][0] == pytest.approx(14.0, rel=1e-14)
        assert res_1["beta102"][0] == pytest.approx(2.0 / 5.0, rel=1e-14)
        assert res_1["isoper"][0] == pytest.approx(4.0 * np.pi * 10.0 / 196.0, rel=1e-14)

        # Length-1 box array
        res_sq = imt_pointpattern([[0.5, 0.5]], box=[3.0])
        assert res_sq["area"][0] == pytest.approx(9.0, rel=1e-14)

    def test_periodic_hexagonal_lattice(self):
        a = 1.0
        h = a * np.sqrt(3.0) / 2.0
        nx, ny = 4, 4
        lx, ly = nx * a, ny * 2.0 * h
        pts = []
        for j in range(ny):
            for i in range(nx):
                pts.append([i * a, j * 2.0 * h])
                pts.append([(i + 0.5) * a, (j + 0.5) * 2.0 * h])
        seeds = np.array(pts)
        res = imt_pointpattern(seeds, box=(lx, ly))

        hex_area = a * h
        hex_peri = 6.0 * (a / np.sqrt(3.0))
        assert np.allclose(res["area"], hex_area, atol=1e-13)
        assert np.allclose(res["perimeter"], hex_peri, atol=1e-13)
        assert np.allclose(res["q"][:, 6], 1.0, atol=1e-13)
        assert np.allclose(res["q"][:, 12], 1.0, atol=1e-13)
        assert np.allclose(res["q"][:, 2], 0.0, atol=1e-13)
        assert np.allclose(res["q"][:, 4], 0.0, atol=1e-13)

    def test_periodic_random_area_conservation_and_shift_invariance(self):
        rng = np.random.default_rng(2026)
        lx, ly = 7.5, 4.2
        seeds = np.column_stack([rng.uniform(0.0, lx, 60), rng.uniform(0.0, ly, 60)])
        res = imt_pointpattern(seeds, box=(lx, ly))

        assert not np.any(np.isnan(res["area"]))
        assert np.sum(res["area"]) == pytest.approx(lx * ly, rel=1e-12)
        assert np.all(res["area"] > 0.0)
        assert np.all(np.abs(res["psi"][:, 1]) < 1e-12)

        shifted = np.column_stack([(seeds[:, 0] + 3.17) % lx, (seeds[:, 1] + 2.41) % ly])
        res_s = imt_pointpattern(shifted, box=(lx, ly))
        assert np.allclose(res["area"], res_s["area"], atol=1e-12)
        assert np.allclose(res["perimeter"], res_s["perimeter"], atol=1e-12)
        assert np.allclose(res["psi"], res_s["psi"], atol=1e-12)

    def test_non_periodic_bounded_and_unbounded(self):
        # 5x5 square lattice: 9 interior cells are exact unit squares, 16 border cells are NaN
        xs, ys = np.meshgrid(np.arange(5, dtype=float), np.arange(5, dtype=float))
        seeds = np.column_stack([xs.ravel(), ys.ravel()])
        res = imt_pointpattern(seeds)

        is_interior = (seeds[:, 0] > 0) & (seeds[:, 0] < 4) & (seeds[:, 1] > 0) & (seeds[:, 1] < 4)
        assert np.allclose(res["area"][is_interior], 1.0, atol=1e-13)
        assert np.allclose(res["perimeter"][is_interior], 4.0, atol=1e-13)
        assert np.allclose(res["q"][is_interior, 4], 1.0, atol=1e-13)

        assert np.all(np.isnan(res["area"][~is_interior]))
        assert np.all(np.isnan(res["perimeter"][~is_interior]))
        assert np.all(np.isnan(res["beta102"][~is_interior]))
        assert np.all(np.isnan(res["isoper"][~is_interior]))
        assert np.all(np.isnan(res["psi"][~is_interior]))
        assert np.all(np.isnan(res["q"][~is_interior]))

    def test_edge_cases_and_validation(self):
        # Empty input
        empty = imt_pointpattern(np.empty((0, 2)))
        assert empty["area"].shape == (0,)
        assert empty["psi"].shape == (0, 13)

        # Fewer than 4 non-periodic points -> all NaN
        small = imt_pointpattern([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        assert np.all(np.isnan(small["area"]))

        # Collinear non-periodic points -> all NaN
        collinear = imt_pointpattern([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        assert np.all(np.isnan(collinear["area"]))

        # Seed outside periodic box
        with pytest.raises(ValueError, match="outside periodic box"):
            imt_pointpattern([[0.5, 1.0]], box=(1.0, 1.0))
        with pytest.raises(ValueError, match="outside periodic box"):
            imt_pointpattern([[-0.1, 0.5]], box=(1.0, 1.0))

        # Invalid box / shape
        with pytest.raises(ValueError):
            imt_pointpattern([[0.5, 0.5]], box=(1.0, -1.0))
        with pytest.raises(ValueError):
            imt_pointpattern([[0.5, 0.5]], box=True)
        with pytest.raises(ValueError):
            imt_pointpattern([[0.5, 0.5]], box=[1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            imt_pointpattern([[0.5, 0.5, 0.5]])
        with pytest.raises(ValueError):
            imt_pointpattern([[np.nan, 0.5]])


# ---------------------------------------------------------------------------
# 6. 2D vs. 3D extruded-prism consistency check
# ---------------------------------------------------------------------------

def _extrude_convex_polygon_ccw(verts_2d, height):
    """Build a closed, outward-oriented 3D triangular prism mesh from a CCW convex 2D polygon."""
    pts = np.asarray(verts_2d, dtype=np.float64)
    n = len(pts)
    bottom = np.column_stack([pts, np.zeros(n)])
    top = np.column_stack([pts, np.full(n, float(height))])
    verts_3d = np.vstack([bottom, top])

    faces = []
    # Side quads (2 triangles per edge, outward normal = edge x +z)
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, n + j])
        faces.append([i, n + j, n + i])
    # Top (+z) and bottom (-z) fan triangulations
    for i in range(1, n - 1):
        faces.append([n, n + i, n + i + 1])
        faces.append([0, i + 1, i])
    return verts_3d, np.asarray(faces, dtype=np.int64)


class Test2D3DCrossCheck:
    @pytest.mark.parametrize(
        "verts_2d,height",
        [
            # Axis-aligned rectangle
            ([[0.0, 0.0], [3.5, 0.0], [3.5, 1.2], [0.0, 1.2]], 2.0),
            # Rotated asymmetric convex pentagon (tests nonzero off-diagonal T_xy / Im(psi_2))
            ([[0.0, 0.0], [2.4, 0.7], [3.1, 2.3], [1.2, 3.0], [-0.4, 1.5]], 1.7),
        ],
    )
    def test_extruded_prism_matches_3d_minkowski_tensors(self, verts_2d, height):
        res_2d = imt_polygon(verts_2d)
        verts_3d, faces_3d = _extrude_convex_polygon_ccw(verts_2d, height)
        res_3d = pk.minkowski_tensors(verts_3d, faces_3d, compute=["w000", "w100", "w102"])

        # Volume: w000 == h * A
        assert res_3d["w000"] == pytest.approx(height * res_2d["area"], rel=1e-13)

        # Surface area: 3 * w100 == 2 * A + h * P
        assert 3.0 * res_3d["w100"] - 2.0 * res_2d["area"] == pytest.approx(
            height * res_2d["perimeter"], rel=1e-13
        )

        # 2x2 xy-subblock of 3D w102 equals (h / 3) * int(n (x) n dl), because top/bottom
        # caps only contribute to zz. In terms of 2D IMT psi_0 and psi_2:
        #   T_xx = (psi_0 + Re(psi_2)) / 2
        #   T_yy = (psi_0 - Re(psi_2)) / 2
        #   T_xy = Im(psi_2) / 2
        psi0 = res_2d["psi"][0].real
        psi2 = res_2d["psi"][2]
        expected_2d_w102 = 0.5 * np.array([
            [psi0 + psi2.real, psi2.imag],
            [psi2.imag, psi0 - psi2.real],
        ])
        xy_block = (3.0 / height) * res_3d["w102"][:2, :2]
        assert np.allclose(xy_block, expected_2d_w102, rtol=1e-13, atol=1e-13)

        # Eigenvalue ratio of xy_block equals 2D beta102
        eigvals = np.linalg.eigvalsh(xy_block)
        assert eigvals[0] / eigvals[1] == pytest.approx(res_2d["beta102"], rel=1e-13)

