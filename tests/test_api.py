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
        actual = self.result['w020_eigvals']
        np.testing.assert_allclose(actual, expected, rtol=1e-4)

    def test_w120_eigenvalues(self):
        a, b, c = self.a, self.b, self.c
        expected = sorted([
            1.0/6 * (1.0/3 * a**3 * (b + c) + a**2 * b * c),
            1.0/6 * (1.0/3 * b**3 * (a + c) + b**2 * a * c),
            1.0/6 * (1.0/3 * c**3 * (a + b) + c**2 * a * b),
        ])
        actual = self.result['w120_eigvals']
        np.testing.assert_allclose(actual, expected, rtol=1e-4)

    def test_w220_eigenvalues(self):
        a, b, c = self.a, self.b, self.c
        pi = math.pi
        expected = sorted([
            pi / 36 * (a**3 + 3 * a**2 * (b + c)),
            pi / 36 * (b**3 + 3 * b**2 * (a + c)),
            pi / 36 * (c**3 + 3 * c**2 * (a + b)),
        ])
        actual = self.result['w220_eigvals']
        np.testing.assert_allclose(actual, expected, rtol=1e-4)

    def test_w320_eigenvalues(self):
        a, b, c = self.a, self.b, self.c
        pi = math.pi
        expected = sorted([pi / 3 * a**2, pi / 3 * b**2, pi / 3 * c**2])
        actual = self.result['w320_eigvals']
        np.testing.assert_allclose(actual, expected, rtol=1e-4)

    def test_w102_eigenvalues(self):
        a, b, c = self.a, self.b, self.c
        expected = sorted([2.0/3 * a * b, 2.0/3 * a * c, 2.0/3 * b * c])
        actual = self.result['w102_eigvals']
        np.testing.assert_allclose(actual, expected, rtol=1e-4)

    def test_w202_eigenvalues(self):
        a, b, c = self.a, self.b, self.c
        pi = math.pi
        expected = sorted([pi / 6 * (a + b), pi / 6 * (a + c), pi / 6 * (b + c)])
        actual = self.result['w202_eigvals']
        np.testing.assert_allclose(actual, expected, rtol=1e-4)

    def test_eigvals_ascending_magnitude(self):
        """#59: eigenvalues are returned in ascending |λ| order."""
        for name in ['w020', 'w120', 'w220', 'w320', 'w102', 'w202']:
            evs = np.abs(self.result[f'{name}_eigvals'])
            assert list(evs) == sorted(evs), \
                f"{name}_eigvals not in ascending magnitude order"

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

    def test_center_wrong_shape_2d_raises(self):
        """#46: center with shape (2,) raises ValueError with helpful message."""
        with pytest.raises(ValueError, match="center must be a \\(3,\\) array"):
            minkowski_functionals(self.verts, self.faces, center=[1.0, 2.0])

    def test_center_wrong_shape_2d_array_raises(self):
        """#46: center with shape (1, 3) raises ValueError."""
        with pytest.raises(ValueError, match="center must be a \\(3,\\) array"):
            minkowski_functionals(self.verts, self.faces, center=[[1.0, 2.0, 3.0]])


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

    def test_compute_unknown_name_raises(self):
        """#45: unknown name in compute list raises ValueError."""
        with pytest.raises(ValueError, match="Unknown compute names"):
            minkowski_functionals(self.verts, self.faces, compute=['w000', 'typo'])

    def test_compute_all_unknown_raises(self):
        """#45: entirely unrecognised list raises ValueError."""
        with pytest.raises(ValueError, match="Unknown compute names"):
            minkowski_functionals(self.verts, self.faces, compute=['bad_name'])

    def test_compute_single_tensor(self):
        result = minkowski_functionals(self.verts, self.faces, compute=['w102'])
        assert 'w102' in result
        assert 'w102_eigvals' in result
        assert 'w102_eigvecs' in result


class TestNumericSafety:
    """Zero-guard fixes: #42 get_ref_vec, #43 angle_sum==0, #49 toroidal w300=0."""

    def test_get_ref_vec_near_zero_scalar_warns_and_returns_zeros(self):
        """#42: get_ref_vec falls back to origin when scalar denominator is near zero."""
        import warnings
        from pykarambola.minkowski import get_ref_vec
        from pykarambola.results import MinkValResult

        w_scalar = {0: MinkValResult(result=0.0)}
        w_vector = {0: MinkValResult(result=np.array([1.0, 2.0, 3.0]))}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = get_ref_vec(0, w_scalar, w_vector, denominator_name='w300')

        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert "near zero" in str(w[0].message).lower()
        assert "w300" in str(w[0].message)
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])

    def test_get_ref_vec_nonzero_scalar_returns_correct_value(self):
        """#42: get_ref_vec works normally when the scalar is non-zero."""
        from pykarambola.minkowski import get_ref_vec
        from pykarambola.results import MinkValResult

        w_scalar = {0: MinkValResult(result=2.0)}
        w_vector = {0: MinkValResult(result=np.array([4.0, 6.0, 8.0]))}
        result = get_ref_vec(0, w_scalar, w_vector)
        np.testing.assert_array_almost_equal(result, [2.0, 3.0, 4.0])

    @pytest.mark.parametrize("fn_name", [
        "calculate_w300",
        "calculate_w310",
        "calculate_w320",
    ])
    def test_zero_angle_sum_no_nan_and_warns(self, fn_name):
        """#43: w300/w310/w320 handle a vertex with angle_sum<=0 without NaN/crash and emit a warning."""
        import warnings
        import pykarambola.minkowski as mink
        from pykarambola.triangulation import Triangulation

        fn = getattr(mink, fn_name)

        verts, faces = _box_mesh(2.0, 3.0, 4.0)
        surf = Triangulation.from_arrays(verts, faces)
        surf._vertex_angle_sums[0] = 0.0  # simulate isolated/degenerate vertex

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = fn(surf)

        assert any(issubclass(warning.category, UserWarning) for warning in w)

        from pykarambola.tensor import SymmetricMatrix3
        for val in result.values():
            r = val.result
            arr = r.to_numpy() if isinstance(r, SymmetricMatrix3) else np.atleast_1d(r)
            assert np.all(np.isfinite(arr.astype(float)))

    def test_w320_centroid_zero_w300_warns_and_falls_back(self):
        """#49: w320 with center='centroid' and w300==0 (torus) falls back to origin."""
        import warnings
        from pykarambola.triangulation import Triangulation
        from pykarambola.minkowski import calculate_w320
        from pykarambola.results import MinkValResult

        verts, faces = _box_mesh(2.0, 3.0, 4.0)
        surf = Triangulation.from_arrays(verts, faces)

        # Simulate w300==0 as occurs for a toroidal surface (Euler characteristic = 0)
        w300_zero = {0: MinkValResult(result=0.0)}
        w310_nonzero = {0: MinkValResult(result=np.array([1.0, 0.0, 0.0]))}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = calculate_w320(surf, w300=w300_zero, w310=w310_nonzero)

        assert any(issubclass(warning.category, UserWarning) for warning in w)
        for i_idx, j_idx in [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)]:
            assert np.isfinite(result[0].result[i_idx, j_idx])


class TestReturnTypeAsymmetry:
    """#48: flat dict vs nested dict depending on labels parameter."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.verts, self.faces = _box_mesh(2.0, 3.0, 4.0)

    def test_no_labels_returns_flat_dict(self):
        result = minkowski_functionals(self.verts, self.faces, labels=None)
        assert 'w000' in result

    def test_labels_provided_returns_nested_dict(self):
        labels = np.zeros(len(self.faces), dtype=int)
        result = minkowski_functionals(self.verts, self.faces, labels=labels)
        assert 0 in result
        assert 'w000' in result[0]
        # Flat key access does not work on the nested result
        assert 'w000' not in result

    def test_labels_single_value_still_nested(self):
        """Even one unique label → nested, not flat."""
        labels = np.ones(len(self.faces), dtype=int)
        result = minkowski_functionals(self.verts, self.faces, labels=labels)
        assert 1 in result
        assert isinstance(result[1], dict)


class TestComputeEigensystems:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.verts, self.faces = _box_mesh(2.0, 3.0, 4.0)

    def test_false_omits_eigvals_and_eigvecs(self):
        result = minkowski_functionals(
            self.verts, self.faces, compute_eigensystems=False,
        )
        for name in ['w020', 'w120', 'w220', 'w320', 'w102', 'w202']:
            assert f'{name}_eigvals' not in result
            assert f'{name}_eigvecs' not in result

    def test_false_retains_tensor_values(self):
        result = minkowski_functionals(
            self.verts, self.faces, compute_eigensystems=False,
        )
        for name in ['w020', 'w120', 'w220', 'w320', 'w102', 'w202']:
            assert name in result
            assert result[name].shape == (3, 3)

    def test_false_works_with_scalar_only_compute(self):
        result = minkowski_functionals(
            self.verts, self.faces,
            compute=['w000', 'w100', 'w200', 'w300'],
            compute_eigensystems=False,
        )
        assert set(result.keys()) == {'w000', 'w100', 'w200', 'w300'}

    def test_true_default_includes_eigvals_and_eigvecs(self):
        result = minkowski_functionals(self.verts, self.faces)
        for name in ['w020', 'w120', 'w220', 'w320', 'w102', 'w202']:
            assert f'{name}_eigvals' in result
            assert f'{name}_eigvecs' in result

    def test_single_rank2_tensor_omits_eigensystem(self):
        # Requesting a single rank-2 tensor with compute_eigensystems=False
        # should return the tensor matrix but suppress its eigvals/eigvecs.
        result = minkowski_functionals(
            self.verts, self.faces,
            compute=['w102'],
            compute_eigensystems=False,
        )
        assert 'w102' in result
        assert result['w102'].shape == (3, 3)
        assert 'w102_eigvals' not in result
        assert 'w102_eigvecs' not in result

    def test_beta_plain_key_raises_unknown(self):
        # Plain 'beta' is not a valid compute name; use 'w020_beta' etc. instead.
        with pytest.raises(ValueError, match="Unknown compute names"):
            minkowski_functionals(
                self.verts, self.faces,
                compute=['w020', 'beta'],
                compute_eigensystems=False,
            )

    def test_beta_suffix_with_false_raises_value_error(self):
        # Same forward-compatibility guard for any *_beta quantity.
        with pytest.raises(ValueError, match="compute_eigensystems=True"):
            minkowski_functionals(
                self.verts, self.faces,
                compute=['w020', 'w020_beta'],
                compute_eigensystems=False,
            )


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

    def test_float_label_image_warns(self):
        """#52: float32 label image triggers a UserWarning and still works."""
        vol = _voxel_box((20, 20, 20), np.s_[5:15, 5:15, 5:15]).astype(np.float32)
        with pytest.warns(UserWarning, match="dtype"):
            result = minkowski_functionals_from_label_image(vol)
        assert 1 in result
        assert 'w000' in result[1]

    def test_integer_label_image_no_dtype_warning(self):
        """#52: integer label image produces no dtype warning."""
        import warnings
        vol = _voxel_box((20, 20, 20), np.s_[5:15, 5:15, 5:15])  # int32
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            minkowski_functionals_from_label_image(vol)
        dtype_warnings = [w for w in caught if "dtype" in str(w.message).lower()]
        assert len(dtype_warnings) == 0

    def test_compute_eigensystems_false_threads_through(self):
        vol = _voxel_box((20, 20, 20), np.s_[5:15, 5:15, 5:15])
        result = minkowski_functionals_from_label_image(
            vol, compute_eigensystems=False,
        )
        for name in ['w020', 'w120', 'w220', 'w320', 'w102', 'w202']:
            assert f'{name}_eigvals' not in result[1]
            assert f'{name}_eigvecs' not in result[1]


class TestDerivedScalars:
    """Tests for beta (#1), traces (#2), and dependency chain (#53)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.verts, self.faces = _box_mesh(2.0, 3.0, 4.0)

    def test_w020_beta_via_derived_dep(self):
        """#53: requesting w020_beta alone auto-promotes w020 into wanted."""
        result = minkowski_functionals(self.verts, self.faces, compute=['w020_beta'])
        assert 'w020_beta' in result
        assert 0.0 <= result['w020_beta'] <= 1.0

    def test_beta_range(self):
        """#1: beta in [0, 1] for all rank-2 tensors on the standard box."""
        result = minkowski_functionals(self.verts, self.faces, compute='all')
        for name in ['w020', 'w120', 'w220', 'w320', 'w102', 'w202']:
            b = result[f'{name}_beta']
            assert 0.0 <= b <= 1.0, f"{name}_beta={b} out of range"

    def test_beta_requires_eigensystems_false_raises(self):
        """#1: requesting beta with compute_eigensystems=False raises ValueError."""
        with pytest.raises(ValueError, match="requires eigensystems"):
            minkowski_functionals(self.verts, self.faces,
                                  compute=['w020_beta'],
                                  compute_eigensystems=False)

    def test_w020_trace(self):
        """#2: w020_trace = Tr(w020) = np.trace of the 3x3 matrix."""
        result = minkowski_functionals(self.verts, self.faces,
                                       compute=['w020', 'w020_trace'])
        expected_trace = float(np.trace(result['w020']))
        assert result['w020_trace'] == pytest.approx(expected_trace, rel=1e-6)

    def test_w020_trace_ratio_via_derived_dep(self):
        """#53: requesting w020_trace_ratio alone auto-promotes w020 and w000."""
        result = minkowski_functionals(self.verts, self.faces,
                                       compute=['w020_trace_ratio'])
        assert 'w020_trace_ratio' in result
        assert np.isfinite(result['w020_trace_ratio'])

    def test_compute_all_includes_derived_keys(self):
        """#53: compute='all' includes all beta, trace, trace_ratio keys."""
        result = minkowski_functionals(self.verts, self.faces, compute='all')
        for name in ['w020', 'w120', 'w220', 'w320', 'w102', 'w202']:
            assert f'{name}_beta' in result
        for name in ['w020', 'w120', 'w220', 'w320', 'w102', 'w202']:
            assert f'{name}_trace' in result
        for name in ['w020', 'w120', 'w220', 'w320']:
            assert f'{name}_trace_ratio' in result


class TestPrerequisiteOptimization:
    """#47: w000/w010 must not be computed when only w020 is wanted (no centroid)."""

    def test_compute_w020_only_no_centroid(self):
        """compute=['w020'] in non-centroid mode must not produce w000/w010."""
        verts, faces = _box_mesh(2.0, 3.0, 4.0)
        result = minkowski_functionals(verts, faces, compute=['w020'])
        assert 'w020' in result
        assert 'w000' not in result
        assert 'w010' not in result

    def test_compute_w020_centroid_includes_prerequisites(self):
        """With center='centroid', w020 is still computed correctly."""
        verts, faces = _box_mesh(2.0, 3.0, 4.0)
        result = minkowski_functionals(verts, faces, compute=['w020'], center='centroid')
        assert 'w020' in result
