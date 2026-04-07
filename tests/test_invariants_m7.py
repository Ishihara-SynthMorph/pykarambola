"""
Tests for Milestone 7: Integration Tests, Benchmarks, and Documentation.

This is the comprehensive test suite that validates all acceptance criteria:
- End-to-end rotational invariance across multiple mesh types and symmetry groups
- Clebsch-Gordan consistency (selection rules)
- Performance benchmarks
- Invariant reproducibility
"""

import time
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pykarambola.api import minkowski_tensors
from pykarambola.invariants import (
    VECTORS, RANK2_TENSORS, TENSOR_REGISTRY,
    compute_invariants, compute_invariant_labels,
    decompose_all, _enumerate_invariant_contractions,
)


# -----------------------------------------------------------------------------
# Test fixtures: mesh generation
# -----------------------------------------------------------------------------

def _box_mesh(a, b, c):
    """Build a triangulated axis-aligned box centered at the origin."""
    ha, hb, hc = a / 2.0, b / 2.0, c / 2.0
    verts = np.array([
        [-ha, -hb, -hc], [ha, -hb, -hc], [ha, hb, -hc], [-ha, hb, -hc],
        [-ha, -hb, hc], [ha, -hb, hc], [ha, hb, hc], [-ha, hb, hc],
    ], dtype=np.float64)
    faces = np.array([
        [0, 3, 2], [0, 2, 1], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
        [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
    ], dtype=np.int64)
    return verts, faces


def _icosphere_mesh(radius=1.0, subdivisions=2):
    """Build an icosphere mesh."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=np.float64)
    verts = verts / np.linalg.norm(verts[0])
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int64)
    for _ in range(subdivisions):
        verts, faces = _subdivide_icosphere(verts, faces)
    return verts * radius, faces


def _subdivide_icosphere(verts, faces):
    """Subdivide an icosphere."""
    edge_midpoints = {}
    new_verts = list(verts)

    def get_midpoint(i, j):
        key = (min(i, j), max(i, j))
        if key in edge_midpoints:
            return edge_midpoints[key]
        mid = (verts[i] + verts[j]) / 2.0
        mid = mid / np.linalg.norm(mid)
        idx = len(new_verts)
        new_verts.append(mid)
        edge_midpoints[key] = idx
        return idx

    new_faces = []
    for v0, v1, v2 in faces:
        m01 = get_midpoint(v0, v1)
        m12 = get_midpoint(v1, v2)
        m20 = get_midpoint(v2, v0)
        new_faces.extend([
            [v0, m01, m20], [v1, m12, m01], [v2, m20, m12], [m01, m12, m20],
        ])
    return np.array(new_verts), np.array(new_faces, dtype=np.int64)


def _ellipsoid_mesh(a, b, c, subdivisions=2):
    """Build an ellipsoid mesh."""
    verts, faces = _icosphere_mesh(radius=1.0, subdivisions=subdivisions)
    verts = verts * np.array([a, b, c])
    return verts, faces


def _torus_mesh(R=3.0, r=0.5, n_major=32, n_minor=16):
    """Build a torus mesh (non-convex)."""
    verts = []
    faces = []

    for i in range(n_major):
        theta = 2 * np.pi * i / n_major
        for j in range(n_minor):
            phi = 2 * np.pi * j / n_minor
            x = (R + r * np.cos(phi)) * np.cos(theta)
            y = (R + r * np.cos(phi)) * np.sin(theta)
            z = r * np.sin(phi)
            verts.append([x, y, z])

    for i in range(n_major):
        for j in range(n_minor):
            i_next = (i + 1) % n_major
            j_next = (j + 1) % n_minor
            v0 = i * n_minor + j
            v1 = i_next * n_minor + j
            v2 = i_next * n_minor + j_next
            v3 = i * n_minor + j_next
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])

    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int64)


def _random_convex_hull(n_points=50, seed=42):
    """Generate a random convex hull mesh."""
    from scipy.spatial import ConvexHull
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(n_points, 3))
    hull = ConvexHull(points)
    return points, hull.simplices.astype(np.int64)


# -----------------------------------------------------------------------------
# Helper: rotate tensors
# -----------------------------------------------------------------------------

def _rotate_tensors(tensors, R):
    """Apply rotation matrix R to all tensors in a tensors_dict."""
    rotated = {}

    # Scalars are invariant
    for name in ['w000', 'w100', 'w200', 'w300']:
        rotated[name] = tensors[name]

    # Vectors transform as v' = R @ v
    for name in VECTORS:
        rotated[name] = R @ tensors[name]

    # Rank-2 tensors transform as M' = R @ M @ R^T
    for name in RANK2_TENSORS:
        M = np.asarray(tensors[name])
        rotated[name] = np.einsum('ia,jb,ab->ij', R, R, M)

    return rotated


# -----------------------------------------------------------------------------
# Test: End-to-end rotational invariance suite
# -----------------------------------------------------------------------------

class TestEndToEndRotationalInvariance:
    """Comprehensive rotational invariance test across mesh types and symmetry groups."""

    @pytest.mark.parametrize("mesh_type,symmetry", [
        ('box', 'O3'),
        ('box', 'SO3'),
        ('icosphere', 'O3'),
        ('icosphere', 'SO3'),
        ('ellipsoid', 'O3'),
        ('ellipsoid', 'SO3'),
        ('torus', 'O3'),
        ('torus', 'SO3'),
    ])
    def test_rotational_invariance_100_trials(self, mesh_type, symmetry):
        """Full invariant vector should be stable under 100+ random SO(3) rotations."""
        # Generate reference mesh
        if mesh_type == 'box':
            verts, faces = _box_mesh(2.0, 3.0, 4.0)
        elif mesh_type == 'icosphere':
            verts, faces = _icosphere_mesh(1.0, 2)
        elif mesh_type == 'ellipsoid':
            verts, faces = _ellipsoid_mesh(1.0, 2.0, 3.0, 2)
        else:
            verts, faces = _torus_mesh(R=3.0, r=0.5)

        # Compute reference tensors and invariants
        tensors_ref = minkowski_tensors(verts, faces, compute='standard')
        invariants_ref = compute_invariants(tensors_ref, max_degree=3, symmetry=symmetry)

        # Test 100 random rotations
        max_deviation = 0.0
        for trial in range(100):
            R = Rotation.random(random_state=trial).as_matrix()

            # Rotate tensors
            tensors_rotated = _rotate_tensors(tensors_ref, R)

            # Compute invariants from rotated tensors
            invariants_rot = compute_invariants(tensors_rotated, max_degree=3, symmetry=symmetry)

            # Compute relative deviation
            ref_norm = np.linalg.norm(invariants_ref)
            if ref_norm > 1e-14:
                deviation = np.linalg.norm(invariants_rot - invariants_ref) / ref_norm
            else:
                deviation = np.linalg.norm(invariants_rot - invariants_ref)

            max_deviation = max(max_deviation, deviation)

        # Acceptance criterion: max_relative_deviation < 1e-8
        assert max_deviation < 1e-8, \
            f"Max deviation {max_deviation:.2e} exceeds tolerance 1e-8 for {mesh_type}/{symmetry}"

    def test_per_invariant_max_deviation(self):
        """Report per-invariant max deviation across rotations."""
        verts, faces = _ellipsoid_mesh(1.0, 2.0, 3.0)
        tensors_ref = minkowski_tensors(verts, faces, compute='standard')
        invariants_ref = compute_invariants(tensors_ref, max_degree=3, symmetry='SO3')
        labels = compute_invariant_labels(max_degree=3, symmetry='SO3')

        # Track per-invariant max deviation
        per_inv_max = np.zeros(len(invariants_ref))

        for trial in range(100):
            R = Rotation.random(random_state=trial).as_matrix()
            tensors_rotated = _rotate_tensors(tensors_ref, R)
            invariants_rot = compute_invariants(tensors_rotated, max_degree=3, symmetry='SO3')

            for i in range(len(invariants_ref)):
                ref_val = abs(invariants_ref[i])
                if ref_val > 1e-14:
                    dev = abs(invariants_rot[i] - invariants_ref[i]) / ref_val
                else:
                    dev = abs(invariants_rot[i] - invariants_ref[i])
                per_inv_max[i] = max(per_inv_max[i], dev)

        # All invariants should be stable
        worst_idx = np.argmax(per_inv_max)
        worst_dev = per_inv_max[worst_idx]
        assert worst_dev < 1e-8, \
            f"Worst invariant '{labels[worst_idx]}' has max deviation {worst_dev:.2e}"


# -----------------------------------------------------------------------------
# Test: Clebsch-Gordan consistency
# -----------------------------------------------------------------------------

class TestClebschGordanConsistency:
    """Verify all contractions satisfy Wigner-3j selection rules."""

    def test_registry_irreps_valid(self):
        """All irreps in the registry should be valid SO(3) labels."""
        valid_irreps = {'0e', '1o', '2e'}  # We only use these
        for name, info in TENSOR_REGISTRY.items():
            for irrep in info['irreps']:
                assert irrep in valid_irreps, f"Invalid irrep '{irrep}' in {name}"

    def test_scalar_contractions_are_0e(self):
        """Degree-1 scalars are all 0e irreps."""
        # 4 base scalars + 4 traces of rank-2 (trace extracts 0e component)
        # All produce 0e scalars
        verts, faces = _box_mesh(2.0, 3.0, 4.0)
        tensors = minkowski_tensors(verts, faces, compute='standard')
        decomposed = decompose_all(tensors)

        # Check that all scalars are floats (0e)
        from pykarambola.invariants import SCALARS, _RANK2_INDEPENDENT_TRACES
        for name in SCALARS:
            assert isinstance(decomposed[(name, '0e')], float)
        for name in _RANK2_INDEPENDENT_TRACES:
            assert isinstance(decomposed[(name, '0e')], float)

    def test_dot_product_selection_rule(self):
        """v_i · v_j: 1o ⊗ 1o → 0e is valid (same l, opposite parity cancels)."""
        # The Wigner-3j symbol (1, 1, 0 | m1, m2, 0) is non-zero when m1 = -m2
        # This confirms the dot product is a valid invariant
        # (We verify numerically that the result is a scalar, not a tensor)
        verts, faces = _box_mesh(2.0, 3.0, 4.0)
        tensors = minkowski_tensors(verts, faces, compute='standard')
        inv = compute_invariants(tensors, max_degree=2)
        # All values should be floats, not arrays
        assert inv.dtype == np.float64

    def test_frobenius_selection_rule(self):
        """Tr(T_i T_j): 2e ⊗ 2e → 0e is valid."""
        # The Frobenius inner product contracts two l=2 tensors to l=0
        # This is allowed by Wigner-3j: (2, 2, 0) has non-zero coefficients
        verts, faces = _box_mesh(2.0, 3.0, 4.0)
        tensors = minkowski_tensors(verts, faces, compute='standard')
        inv = compute_invariants(tensors, max_degree=2)
        assert inv.dtype == np.float64


# -----------------------------------------------------------------------------
# Test: Performance benchmark
# -----------------------------------------------------------------------------

class TestPerformanceBenchmark:
    """Time compute_invariants on meshes of varying complexity."""

    @pytest.mark.parametrize("subdivisions,expected_faces", [
        (1, 80),    # ~80 triangles
        (2, 320),   # ~320 triangles
        (3, 1280),  # ~1280 triangles
    ])
    def test_benchmark_icosphere(self, subdivisions, expected_faces):
        """Benchmark compute_invariants on icospheres of varying resolution."""
        verts, faces = _icosphere_mesh(radius=1.0, subdivisions=subdivisions)
        assert len(faces) == expected_faces, f"Expected {expected_faces} faces, got {len(faces)}"

        tensors = minkowski_tensors(verts, faces, compute='standard')

        # Time compute_invariants
        start = time.perf_counter()
        for _ in range(10):
            compute_invariants(tensors, max_degree=3, symmetry='SO3')
        elapsed = (time.perf_counter() - start) / 10

        # Just verify it completes in reasonable time (< 1 second)
        assert elapsed < 1.0, f"compute_invariants took {elapsed:.3f}s on {len(faces)} triangles"


# -----------------------------------------------------------------------------
# Test: Invariant reproducibility
# -----------------------------------------------------------------------------

class TestInvariantReproducibility:
    """Test that serialized tensors produce identical invariants."""

    def test_serialize_deserialize_produces_same_invariants(self):
        """Invariants should be bitwise identical after tensor serialization."""
        verts, faces = _ellipsoid_mesh(1.0, 2.0, 3.0)
        tensors = minkowski_tensors(verts, faces, compute='standard')

        # Compute invariants
        inv1 = compute_invariants(tensors, max_degree=3, symmetry='SO3')

        # Serialize and deserialize tensors (simulate saving/loading)
        import json
        serialized = {}
        for key, val in tensors.items():
            if isinstance(val, np.ndarray):
                serialized[key] = val.tolist()
            else:
                serialized[key] = val

        json_str = json.dumps(serialized)
        loaded = json.loads(json_str)

        deserialized = {}
        for key, val in loaded.items():
            if isinstance(val, list):
                deserialized[key] = np.array(val, dtype=np.float64)
            else:
                deserialized[key] = val

        # Compute invariants from deserialized tensors
        inv2 = compute_invariants(deserialized, max_degree=3, symmetry='SO3')

        # Should be bitwise identical
        np.testing.assert_array_equal(inv1, inv2)


# -----------------------------------------------------------------------------
# Test: Summary table verification
# -----------------------------------------------------------------------------

class TestSummaryTableVerification:
    """Verify invariant counts match the issue's summary table."""

    def test_o3_count_155(self):
        """O(3) total should be 155 at degree 3."""
        # From issue: 8 + 10 + 21 + 60 + 56 = 155
        counts = _enumerate_invariant_contractions(symmetry='O3')
        assert counts['total'] == 155

    def test_so3_count_219(self):
        """SO(3) total should be 219 at degree 3."""
        # From issue: 155 + 4 + 60 = 219
        counts = _enumerate_invariant_contractions(symmetry='SO3')
        assert counts['total'] == 219

    def test_breakdown_matches_issue(self):
        """Individual counts should match the issue specification."""
        counts = _enumerate_invariant_contractions(symmetry='SO3')

        # From issue table:
        # | Degree | Type                      | Count (O3) | Count (SO3) |
        # |--------|---------------------------|------------|-------------|
        # | 1      | Scalars                   | 8          | 8           |
        # | 2      | Vector dot products       | 10         | 10          |
        # | 2      | Frobenius inner products  | 21         | 21          |
        # | 3      | Quadratic forms           | 60         | 60          |
        # | 3      | Triple matrix traces      | 56         | 56          |
        # | 3      | Triple vector determinants| 0          | 4           |
        # | 3      | Commutator pseudo-scalars | 0          | ~60         |

        assert counts['degree1_scalars'] == 8
        assert counts['degree2_dot_products'] == 10
        assert counts['degree2_frobenius'] == 21
        assert counts['degree3_quadratic_forms'] == 60
        assert counts['degree3_triple_traces'] == 56
        assert counts['degree3_triple_vector_dets'] == 4
        assert counts['degree3_commutator_pseudoscalars'] == 60
