"""
Tests for the high-level minkowski_functionals() API.
Uses a box mesh (a=2, b=3, c=4) built as numpy arrays (no file I/O).
"""

import math
import numpy as np
import pytest

from pykarambola.api import minkowski_functionals, minkowski_functionals_from_label_image


def _box_mesh(a, b, c):
    """Build a triangulated axis-aligned box centered at the origin.

    Returns (verts, faces) numpy arrays. The box spans
    [-a/2, a/2] x [-b/2, b/2] x [-c/2, c/2].
    Each face of the box is split into 2 triangles (12 triangles total).
    Normals point outward.
    """
    ha, hb, hc = a / 2.0, b / 2.0, c / 2.0
    verts = np.array([
        [-ha, -hb, -hc],  # 0
        [ ha, -hb, -hc],  # 1
        [ ha,  hb, -hc],  # 2
        [-ha,  hb, -hc],  # 3
        [-ha, -hb,  hc],  # 4
        [ ha, -hb,  hc],  # 5
        [ ha,  hb,  hc],  # 6
        [-ha,  hb,  hc],  # 7
    ], dtype=np.float64)

    # Each face: 2 triangles, winding for outward normals
    faces = np.array([
        # -z face (normal -z): 0,3,2 and 0,2,1
        [0, 3, 2], [0, 2, 1],
        # +z face (normal +z): 4,5,6 and 4,6,7
        [4, 5, 6], [4, 6, 7],
        # -y face (normal -y): 0,1,5 and 0,5,4
        [0, 1, 5], [0, 5, 4],
        # +y face (normal +y): 2,3,7 and 2,7,6
        [2, 3, 7], [2, 7, 6],
        # -x face (normal -x): 0,4,7 and 0,7,3
        [0, 4, 7], [0, 7, 3],
        # +x face (normal +x): 1,2,6 and 1,6,5
        [1, 2, 6], [1, 6, 5],
    ], dtype=np.int64)

    return verts, faces


class TestStandardBox:
    """Test standard functionals on centered box a=2, b=3, c=4."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.a, self.b, self.c = 2.0, 3.0, 4.0
        self.verts, self.faces = _box_mesh(self.a, self.b, self.c)
        self.result = minkowski_functionals(self.verts, self.faces)

    def test_w000_volume(self):
        expected = self.a * self.b * self.c  # 24
        assert self.result['w000'] == pytest.approx(expected, rel=1e-4)

    def test_w100_area(self):
        expected = 2.0 / 3.0 * (self.a * self.b + self.a * self.c + self.b * self.c)
        assert self.result['w100'] == pytest.approx(expected, rel=1e-4)

    def test_w200_mean_curvature(self):
        expected = math.pi / 3.0 * (self.a + self.b + self.c)
        assert self.result['w200'] == pytest.approx(expected, rel=1e-4)

    def test_w300_euler(self):
        expected = 4.0 * math.pi / 3.0
        assert self.result['w300'] == pytest.approx(expected, rel=1e-4)

    def test_vectors_zero(self):
        for name in ['w010', 'w110', 'w210', 'w310']:
            np.testing.assert_allclose(self.result[name], [0, 0, 0], atol=1e-3,
                                       err_msg=f"{name} should be zero for centered box")

    def test_w020_eigenvalues(self):
        a, b, c = self.a, self.b, self.c
        expected = sorted([a**3 * b * c / 12, b**3 * a * c / 12, c**3 * a * b / 12])
        actual = sorted(self.result['w020_eigvals'])
        np.testing.assert_allclose(actual, expected, rtol=1e-4)

    def test_w120_eigenvalues(self):
        a, b, c = self.a, self.b, self.c
        expected = sorted([
            1.0/6 * (1.0/3 * a**3 * (b + c) + a**2 * b * c),
            1.0/6 * (1.0/3 * b**3 * (a + c) + b**2 * a * c),
            1.0/6 * (1.0/3 * c**3 * (a + b) + c**2 * a * b),
        ])
        actual = sorted(self.result['w120_eigvals'])
        np.testing.assert_allclose(actual, expected, rtol=1e-4)

    def test_w220_eigenvalues(self):
        a, b, c = self.a, self.b, self.c
        pi = math.pi
        expected = sorted([
            pi / 36 * (a**3 + 3 * a**2 * (b + c)),
            pi / 36 * (b**3 + 3 * b**2 * (a + c)),
            pi / 36 * (c**3 + 3 * c**2 * (a + b)),
        ])
        actual = sorted(self.result['w220_eigvals'])
        np.testing.assert_allclose(actual, expected, rtol=1e-4)

    def test_w320_eigenvalues(self):
        a, b, c = self.a, self.b, self.c
        pi = math.pi
        expected = sorted([pi / 3 * a**2, pi / 3 * b**2, pi / 3 * c**2])
        actual = sorted(self.result['w320_eigvals'])
        np.testing.assert_allclose(actual, expected, rtol=1e-4)

    def test_w102_eigenvalues(self):
        a, b, c = self.a, self.b, self.c
        expected = sorted([2.0/3 * a * b, 2.0/3 * a * c, 2.0/3 * b * c])
        actual = sorted(self.result['w102_eigvals'])
        np.testing.assert_allclose(actual, expected, rtol=1e-4)

    def test_w202_eigenvalues(self):
        a, b, c = self.a, self.b, self.c
        pi = math.pi
        expected = sorted([pi / 6 * (a + b), pi / 6 * (a + c), pi / 6 * (b + c)])
        actual = sorted(self.result['w202_eigvals'])
        np.testing.assert_allclose(actual, expected, rtol=1e-4)

    def test_eigvecs_shape(self):
        for name in ['w020', 'w120', 'w220', 'w320', 'w102', 'w202']:
            assert self.result[f'{name}_eigvecs'].shape == (3, 3)

    def test_standard_keys(self):
        expected_keys = {
            'w000', 'w100', 'w200', 'w300',
            'w010', 'w110', 'w210', 'w310',
            'w020', 'w120', 'w220', 'w320', 'w102', 'w202',
        }
        # Plus eigvals/eigvecs for each rank-2
        for name in ['w020', 'w120', 'w220', 'w320', 'w102', 'w202']:
            expected_keys.add(f'{name}_eigvals')
            expected_keys.add(f'{name}_eigvecs')
        assert set(self.result.keys()) == expected_keys


class TestCenterOptions:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.a, self.b, self.c = 2.0, 3.0, 4.0
        self.verts, self.faces = _box_mesh(self.a, self.b, self.c)

    def test_centroid_runs(self):
        result = minkowski_functionals(self.verts, self.faces, center='centroid')
        assert 'w000' in result

    def test_explicit_center_scalars(self):
        shift = np.array([1.0, 2.0, 3.0])
        shifted_verts = self.verts + shift
        result = minkowski_functionals(shifted_verts, self.faces, center=shift)
        # Scalars should match centered box
        expected_vol = self.a * self.b * self.c
        assert result['w000'] == pytest.approx(expected_vol, rel=1e-4)
        expected_area = 2.0 / 3.0 * (self.a * self.b + self.a * self.c + self.b * self.c)
        assert result['w100'] == pytest.approx(expected_area, rel=1e-4)

    def test_explicit_center_vectors_zero(self):
        shift = np.array([1.0, 2.0, 3.0])
        shifted_verts = self.verts + shift
        result = minkowski_functionals(shifted_verts, self.faces, center=shift)
        # Vectors should be zero for re-centered box
        for name in ['w010', 'w110', 'w210', 'w310']:
            np.testing.assert_allclose(result[name], [0, 0, 0], atol=1e-3)


class TestComputeOptions:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.verts, self.faces = _box_mesh(2.0, 3.0, 4.0)

    def test_compute_all_has_extras(self):
        result = minkowski_functionals(self.verts, self.faces, compute='all')
        assert 'w103' in result
        assert 'w104' in result
        assert 'msm_ql' in result
        assert 'msm_wl' in result
        assert result['w103'].shape == (3, 3, 3)
        assert result['w104'].shape == (6, 6)
        assert result['msm_ql'].shape[0] > 0
        assert result['msm_wl'].shape[0] > 0

    def test_compute_subset(self):
        result = minkowski_functionals(self.verts, self.faces, compute=['w000', 'w100'])
        assert set(result.keys()) == {'w000', 'w100'}

    def test_compute_single_tensor(self):
        result = minkowski_functionals(self.verts, self.faces, compute=['w102'])
        assert 'w102' in result
        assert 'w102_eigvals' in result
        assert 'w102_eigvecs' in result


class TestMultiLabel:

    def test_labels_returns_per_label_dict(self):
        verts, faces = _box_mesh(2.0, 3.0, 4.0)
        # Split faces into two groups
        labels = np.array([0]*6 + [1]*6, dtype=np.int64)
        result = minkowski_functionals(verts, faces, labels=labels)
        assert isinstance(result, dict)
        assert 0 in result
        assert 1 in result
        assert 'w000' in result[0]
        assert 'w000' in result[1]
        # Total volume should equal box volume
        total_vol = result[0]['w000'] + result[1]['w000']
        assert total_vol == pytest.approx(2.0 * 3.0 * 4.0, rel=1e-4)


def _voxel_box(shape, box_slices):
    """Create a 3D label image with a box region set to 1."""
    vol = np.zeros(shape, dtype=np.int32)
    vol[box_slices] = 1
    return vol


try:
    import skimage  # noqa: F401
    _has_skimage = True
except ImportError:
    _has_skimage = False


@pytest.mark.skipif(not _has_skimage, reason='scikit-image not installed')
class TestLabelImage:
    """Tests for minkowski_functionals_from_label_image()."""

    def test_single_label_returns_dict_keyed_by_label(self):
        vol = _voxel_box((20, 20, 20), np.s_[5:15, 5:15, 5:15])
        result = minkowski_functionals_from_label_image(vol)
        assert isinstance(result, dict)
        assert 1 in result
        assert 'w000' in result[1]

    def test_voxel_box_volume(self):
        # 10x10x10 voxel box at unit spacing → volume ~ 1000
        vol = _voxel_box((20, 20, 20), np.s_[5:15, 5:15, 5:15])
        result = minkowski_functionals_from_label_image(vol)
        assert result[1]['w000'] == pytest.approx(1000.0, rel=0.05)

    def test_spacing_scales_volume(self):
        vol = _voxel_box((20, 20, 20), np.s_[5:15, 5:15, 5:15])
        r1 = minkowski_functionals_from_label_image(vol, spacing=(1, 1, 1))
        r2 = minkowski_functionals_from_label_image(vol, spacing=(2, 2, 2))
        # Volume scales by 2^3 = 8
        assert r2[1]['w000'] == pytest.approx(8.0 * r1[1]['w000'], rel=0.01)

    def test_spacing_scales_area(self):
        vol = _voxel_box((20, 20, 20), np.s_[5:15, 5:15, 5:15])
        r1 = minkowski_functionals_from_label_image(vol, spacing=(1, 1, 1))
        r2 = minkowski_functionals_from_label_image(vol, spacing=(2, 2, 2))
        # Area (w100) scales by 2^2 = 4
        assert r2[1]['w100'] == pytest.approx(4.0 * r1[1]['w100'], rel=0.01)

    def test_anisotropic_spacing_centroid(self):
        # With non-isotropic spacing, both centroid methods should still centre a
        # symmetric box (vectors near zero) and volume should reflect physical size.
        vol = _voxel_box((20, 20, 20), np.s_[5:15, 5:15, 5:15])
        spacing = (1.0, 2.0, 3.0)
        for method in ('centroid_mesh', 'centroid_voxel'):
            result = minkowski_functionals_from_label_image(vol, spacing=spacing,
                                                            center=method)
            for name in ['w010', 'w110', 'w210', 'w310']:
                np.testing.assert_allclose(
                    result[1][name], [0, 0, 0], atol=0.5,
                    err_msg=f"{name} should be near zero with {method} and anisotropic spacing",
                )
        # Volume should equal 10*10*10 * sz*sy*sx = 1000 * 6
        result = minkowski_functionals_from_label_image(vol, spacing=spacing)
        assert result[1]['w000'] == pytest.approx(6000.0, rel=0.05)

    def test_multi_label(self):
        vol = np.zeros((30, 30, 30), dtype=np.int32)
        vol[2:8, 2:8, 2:8] = 1
        vol[15:25, 15:25, 15:25] = 2
        result = minkowski_functionals_from_label_image(vol)
        assert 1 in result
        assert 2 in result
        # Label 1: 6^3=216, Label 2: 10^3=1000
        assert result[1]['w000'] == pytest.approx(216.0, rel=0.1)
        assert result[2]['w000'] == pytest.approx(1000.0, rel=0.05)

    def test_centroid_default_symmetric_vectors(self):
        # Symmetric box → vectors should be near zero with centroid centering
        vol = _voxel_box((20, 20, 20), np.s_[5:15, 5:15, 5:15])
        result = minkowski_functionals_from_label_image(vol, center='centroid_mesh')
        for name in ['w010', 'w110', 'w210', 'w310']:
            np.testing.assert_allclose(
                result[1][name], [0, 0, 0], atol=0.5,
                err_msg=f"{name} should be near zero for centered symmetric box",
            )

    def test_ascent_gives_positive_signed_vol(self):
        # gradient_direction='ascent' should produce outward normals (positive signed vol)
        from skimage.measure import marching_cubes
        vol = _voxel_box((20, 20, 20), np.s_[5:15, 5:15, 5:15])
        mask = (vol == 1).astype(np.float64)
        verts, faces, _, _ = marching_cubes(
            mask, level=0.5, spacing=(1.0, 1.0, 1.0), gradient_direction='ascent',
        )
        v0 = verts[faces[:, 0]]
        cross = np.cross(verts[faces[:, 1]] - v0, verts[faces[:, 2]] - v0)
        signed_vol = np.sum(v0 * cross) / 6.0
        assert signed_vol > 0, f"Expected positive signed volume, got {signed_vol}"

    def test_centroid_mesh_offcenter_box_near_zero_vectors(self):
        # Off-center symmetric box: mesh-volume centroid should centre the functionals
        vol = _voxel_box((30, 30, 30), np.s_[10:20, 10:20, 10:20])
        result = minkowski_functionals_from_label_image(vol, center='centroid_mesh')
        for name in ['w010', 'w110', 'w210', 'w310']:
            np.testing.assert_allclose(
                result[1][name], [0, 0, 0], atol=0.5,
                err_msg=f"{name} should be near zero with centroid_mesh",
            )

    def test_centroid_voxel_offcenter_box_near_zero_vectors(self):
        # Off-center symmetric box: voxel centroid should also centre the functionals
        vol = _voxel_box((30, 30, 30), np.s_[10:20, 10:20, 10:20])
        result = minkowski_functionals_from_label_image(vol, center='centroid_voxel')
        for name in ['w010', 'w110', 'w210', 'w310']:
            np.testing.assert_allclose(
                result[1][name], [0, 0, 0], atol=0.5,
                err_msg=f"{name} should be near zero with centroid_voxel",
            )

    def test_center_none(self):
        vol = _voxel_box((20, 20, 20), np.s_[5:15, 5:15, 5:15])
        result = minkowski_functionals_from_label_image(vol, center=None)
        assert 'w000' in result[1]
        # With origin as center, vectors should NOT be zero (box not at origin)
        w010_norm = np.linalg.norm(result[1]['w010'])
        assert w010_norm > 1.0

    def test_zero_labels_ignored(self):
        vol = np.zeros((10, 10, 10), dtype=np.int32)
        result = minkowski_functionals_from_label_image(vol)
        assert len(result) == 0
