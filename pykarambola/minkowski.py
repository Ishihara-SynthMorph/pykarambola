"""
All Minkowski functional calculations — vectorized with NumPy.

Each function returns a dict mapping label -> MinkValResult.
"""

import warnings

import numpy as np
from .triangulation import LABEL_UNASSIGNED, NEIGHBOUR_UNASSIGNED
from .results import MinkValResult
from .tensor import SymmetricMatrix3, Rank3Tensor, SymmetricRank4Tensor


def _group_by_label(labels, unique_labels=None):
    """Return {label: boolean_mask} for each unique label."""
    if unique_labels is None:
        unique_labels = np.unique(labels)
    return {int(lab): labels == lab for lab in unique_labels}


def _default_scalar():
    return MinkValResult(result=0.0)


def _default_vector():
    return MinkValResult(result=np.zeros(3, dtype=np.float64))


def _default_matrix():
    return MinkValResult(result=SymmetricMatrix3())


def _default_rank3():
    return MinkValResult(result=Rank3Tensor())


def _default_rank4():
    return MinkValResult(result=SymmetricRank4Tensor())


def get_ref_vec(label, w_scalar, w_vector, eps=1e-12, denominator_name=None):
    """Compute reference vector (centroid) = w_vector / w_scalar for a label.

    Falls back to the origin and emits a warning when the scalar denominator
    is near zero (e.g. open surfaces, flat surfaces, or toroidal shapes where
    the Euler characteristic is zero).
    """
    s = w_scalar[label].result
    if abs(s) < eps:
        hint = f" ({denominator_name})" if denominator_name else ""
        warnings.warn(
            f"Scalar denominator{hint} near zero for label {label}; "
            "falling back to origin reference.",
            stacklevel=3,
        )
        return np.zeros(3)
    return w_vector[label].result / s


# ============================================================================
# Scalar functionals
# ============================================================================

def calculate_w000(surface):
    """Volume via divergence theorem: (1/3) * sum(com . normal * area)."""
    com_dot_n = np.sum(surface._coms * surface._normals, axis=1)  # (F,)
    vals = com_dot_n * surface._areas / 3.0  # (F,)

    results = {}
    for lab, mask in _group_by_label(surface._labels).items():
        r = _default_scalar()
        r.result = float(np.sum(vals[mask]))
        results[lab] = r
    return results


def calculate_w100(surface):
    """Surface area: (1/3) * sum(area)."""
    vals = surface._areas / 3.0
    results = {}
    for lab, mask in _group_by_label(surface._labels).items():
        r = _default_scalar()
        r.result = float(np.sum(vals[mask]))
        results[lab] = r
    return results


def calculate_w200(surface):
    """Mean curvature integral via dihedral angles."""
    # sum over edges: alpha * edge_length / 12
    vals = surface._dihedral_angles * surface._edge_lengths / 12.0  # (F, 3)
    per_tri = np.sum(vals, axis=1)  # (F,)

    results = {}
    for lab, mask in _group_by_label(surface._labels).items():
        r = _default_scalar()
        r.result = float(np.sum(per_tri[mask]))
        results[lab] = r
    return results


def calculate_w300(surface):
    """Gaussian curvature (Euler characteristic) via angle deficit."""
    F = surface.n_triangles()

    # For each triangle vertex (j=0,1,2), compute the angle deficit contribution:
    # w3_part = (2*pi * angle / angle_sum) - angle
    # Each vertex contributes w3_part / 3.0 to the label of the triangle

    # Guard: vertices with zero angle sum (isolated or fully degenerate triangles)
    # contribute nothing to the Gaussian curvature integral.
    if np.any(surface._vertex_angle_sums <= 0):
        warnings.warn(
            "Mesh contains vertices with zero or negative angle sum (isolated or "
            "degenerate triangles); their Gaussian curvature contribution is set to zero.",
            stacklevel=2,
        )

    per_tri = np.zeros(F, dtype=np.float64)
    for j in range(3):
        vert_idx = surface._faces[:, j]  # (F,)
        angle = surface._vertex_angles[:, j]  # (F,)
        angle_sum = surface._vertex_angle_sums[vert_idx]  # (F,)
        safe_angle_sum = np.where(angle_sum > 0, angle_sum, 1.0)
        w3_part = np.where(
            angle_sum > 0,
            2.0 * np.pi * (angle / safe_angle_sum) - angle,
            0.0,
        )
        per_tri += w3_part / 3.0

    results = {}
    for lab, mask in _group_by_label(surface._labels).items():
        r = _default_scalar()
        r.result = float(np.sum(per_tri[mask]))
        results[lab] = r
    return results


# ============================================================================
# Vector functionals
# ============================================================================

def calculate_w010(surface):
    """Volume integral of position (first moment)."""
    c1 = surface._verts[surface._faces[:, 0]]  # (F, 3)
    c2 = surface._verts[surface._faces[:, 1]]
    c3 = surface._verts[surface._faces[:, 2]]
    v = c2 - c1
    w = c3 - c1
    cross_vw = np.cross(v, w)  # (F, 3)

    result_vals = np.zeros((len(surface._faces), 3), dtype=np.float64)
    for i in range(3):
        ip1 = (i + 1) % 3
        part1 = (2 * v[:, i] * v[:, ip1]
                 + v[:, ip1] * w[:, i]
                 + 4 * c1[:, ip1] * (v[:, i] + w[:, i])
                 + v[:, i] * w[:, ip1]
                 + 2 * w[:, i] * w[:, ip1]
                 + 4 * c1[:, i] * (3 * c1[:, ip1] + v[:, ip1] + w[:, ip1]))
        vf = cross_vw[:, ip1]
        result_vals[:, i] = vf * part1 / 24.0

    results = {}
    for lab, mask in _group_by_label(surface._labels).items():
        r = _default_vector()
        r.result = np.sum(result_vals[mask], axis=0)
        results[lab] = r
    return results


def calculate_w110(surface):
    """Surface-weighted position: (1/3) * sum(com * area)."""
    vals = surface._coms * (surface._areas[:, None] / 3.0)  # (F, 3)

    results = {}
    for lab, mask in _group_by_label(surface._labels).items():
        r = _default_vector()
        r.result = np.sum(vals[mask], axis=0)
        results[lab] = r
    return results


def calculate_w210(surface):
    """Curvature-weighted position via dihedral angles."""
    # For each edge j of each triangle: alpha * edge_length * (v_j + v_{j+1}) / 24
    F = surface.n_triangles()
    result_vals = np.zeros((F, 3), dtype=np.float64)

    for j in range(3):
        j1 = (j + 1) % 3
        e_c1 = surface._verts[surface._faces[:, j]]   # (F, 3)
        e_c2 = surface._verts[surface._faces[:, j1]]   # (F, 3)
        alpha = surface._dihedral_angles[:, j]          # (F,)
        e_len = surface._edge_lengths[:, j]             # (F,)
        coeff = alpha * e_len / 24.0                    # (F,)
        result_vals += coeff[:, None] * (e_c1 + e_c2)

    results = {}
    for lab, mask in _group_by_label(surface._labels).items():
        r = _default_vector()
        r.result = np.sum(result_vals[mask], axis=0)
        results[lab] = r
    return results


def calculate_w310(surface):
    """Gaussian curvature-weighted position via angle deficit."""
    if np.any(surface._vertex_angle_sums <= 0):
        warnings.warn(
            "Mesh contains vertices with zero or negative angle sum (isolated or "
            "degenerate triangles); their contribution is set to zero.",
            stacklevel=2,
        )

    F = surface.n_triangles()
    result_vals = np.zeros((F, 3), dtype=np.float64)

    for j in range(3):
        vert_idx = surface._faces[:, j]
        angle = surface._vertex_angles[:, j]
        angle_sum = surface._vertex_angle_sums[vert_idx]
        pos = surface._verts[vert_idx]  # (F, 3)
        safe_angle_sum = np.where(angle_sum > 0, angle_sum, 1.0)
        w310_part = np.where(
            angle_sum > 0,
            (2.0 * np.pi * (angle / safe_angle_sum) - angle) / 3.0,
            0.0,
        )
        result_vals += w310_part[:, None] * pos

    results = {}
    for lab, mask in _group_by_label(surface._labels).items():
        r = _default_vector()
        r.result = np.sum(result_vals[mask], axis=0)
        results[lab] = r
    return results


# ============================================================================
# Matrix functionals
# ============================================================================

# Symmetric matrix index pairs: (0,0), (1,0), (1,1), (2,0), (2,1), (2,2)
_SYM_PAIRS = [(i, j) for i in range(3) for j in range(i + 1)]


def calculate_w020(surface, w000=None, w010=None):
    """Second moment (inertia) tensor."""
    results = {}
    label_groups = _group_by_label(surface._labels)

    # Compute ref_vecs per label
    ref_vecs = {}
    for lab in label_groups:
        if w000 and len(w000) > 0 and lab in w000:
            ref_vecs[lab] = get_ref_vec(lab, w000, w010, denominator_name='w000')
        else:
            ref_vecs[lab] = np.zeros(3)

    for lab, mask in label_groups.items():
        r = _default_matrix()
        ref_vec = ref_vecs[lab]

        c1 = surface._verts[surface._faces[mask, 0]] - ref_vec  # (N, 3)
        c2 = surface._verts[surface._faces[mask, 1]] - ref_vec
        c3 = surface._verts[surface._faces[mask, 2]] - ref_vec
        n = surface._normals[mask]
        area = surface._areas[mask]
        two_area = 2.0 * area

        # Ixx: x=0, z=2
        Ixx = (3 * c2[:, 0]**2 * c2[:, 2] + 2 * c2[:, 0] * c2[:, 2] * c3[:, 0]
               + c2[:, 2] * c3[:, 0]**2
               + c1[:, 2] * (c2[:, 0]**2 + c2[:, 0] * c3[:, 0] + c3[:, 0]**2)
               + c2[:, 0]**2 * c3[:, 2] + 2 * c2[:, 0] * c3[:, 0] * c3[:, 2]
               + 3 * c3[:, 0]**2 * c3[:, 2]
               + c1[:, 0]**2 * (3 * c1[:, 2] + c2[:, 2] + c3[:, 2])
               + c1[:, 0] * (2 * c1[:, 2] * (c2[:, 0] + c3[:, 0])
                              + c2[:, 0] * (2 * c2[:, 2] + c3[:, 2])
                              + c3[:, 0] * (c2[:, 2] + 2 * c3[:, 2]))) / 60.0
        r.result[0, 0] = float(np.sum(Ixx * two_area * n[:, 2]))

        # Iyy: y=1, z=2
        Iyy = (3 * c2[:, 1]**2 * c2[:, 2] + 2 * c2[:, 1] * c2[:, 2] * c3[:, 1]
               + c2[:, 2] * c3[:, 1]**2
               + c1[:, 2] * (c2[:, 1]**2 + c2[:, 1] * c3[:, 1] + c3[:, 1]**2)
               + c2[:, 1]**2 * c3[:, 2] + 2 * c2[:, 1] * c3[:, 1] * c3[:, 2]
               + 3 * c3[:, 1]**2 * c3[:, 2]
               + c1[:, 1]**2 * (3 * c1[:, 2] + c2[:, 2] + c3[:, 2])
               + c1[:, 1] * (2 * c1[:, 2] * (c2[:, 1] + c3[:, 1])
                              + c2[:, 1] * (2 * c2[:, 2] + c3[:, 2])
                              + c3[:, 1] * (c2[:, 2] + 2 * c3[:, 2]))) / 60.0
        r.result[1, 1] = float(np.sum(Iyy * two_area * n[:, 2]))

        # Izz
        Izz = (3 * c2[:, 1] * c2[:, 2]**2 + c2[:, 2]**2 * c3[:, 1]
               + c1[:, 2]**2 * (c2[:, 1] + c3[:, 1])
               + 2 * c2[:, 1] * c2[:, 2] * c3[:, 2] + 2 * c2[:, 2] * c3[:, 1] * c3[:, 2]
               + c2[:, 1] * c3[:, 2]**2 + 3 * c3[:, 1] * c3[:, 2]**2
               + c1[:, 1] * (3 * c1[:, 2]**2 + c2[:, 2]**2 + c2[:, 2] * c3[:, 2]
                              + c3[:, 2]**2 + 2 * c1[:, 2] * (c2[:, 2] + c3[:, 2]))
               + c1[:, 2] * (c2[:, 1] * (2 * c2[:, 2] + c3[:, 2])
                              + c3[:, 1] * (c2[:, 2] + 2 * c3[:, 2]))) / 60.0
        r.result[2, 2] = float(np.sum(Izz * two_area * n[:, 1]))

        # Ixy
        Ixy = (2 * c1[:, 2] * c2[:, 0] * c2[:, 1] + 6 * c2[:, 0] * c2[:, 1] * c2[:, 2]
               + c1[:, 2] * c2[:, 1] * c3[:, 0] + 2 * c2[:, 1] * c2[:, 2] * c3[:, 0]
               + c1[:, 2] * c2[:, 0] * c3[:, 1] + 2 * c2[:, 0] * c2[:, 2] * c3[:, 1]
               + 2 * c1[:, 2] * c3[:, 0] * c3[:, 1] + 2 * c2[:, 2] * c3[:, 0] * c3[:, 1]
               + 2 * c2[:, 0] * c2[:, 1] * c3[:, 2] + 2 * c2[:, 1] * c3[:, 0] * c3[:, 2]
               + 2 * c2[:, 0] * c3[:, 1] * c3[:, 2] + 6 * c3[:, 0] * c3[:, 1] * c3[:, 2]
               + c1[:, 0] * (2 * c2[:, 1] * c2[:, 2] + c2[:, 2] * c3[:, 1]
                              + 2 * c1[:, 2] * (c2[:, 1] + c3[:, 1])
                              + c2[:, 1] * c3[:, 2] + 2 * c3[:, 1] * c3[:, 2]
                              + 2 * c1[:, 1] * (3 * c1[:, 2] + c2[:, 2] + c3[:, 2]))
               + c1[:, 1] * (2 * c1[:, 2] * (c2[:, 0] + c3[:, 0])
                              + c2[:, 0] * (2 * c2[:, 2] + c3[:, 2])
                              + c3[:, 0] * (c2[:, 2] + 2 * c3[:, 2]))) / 120.0

        r.result[0, 1] = float(np.sum(Ixy * two_area * n[:, 2]))
        r.result[0, 2] = float(np.sum(Ixy * two_area * n[:, 1]))
        r.result[1, 2] = float(np.sum(Ixy * two_area * n[:, 0]))

        results[lab] = r

    return results


def calculate_w120(surface, w100=None, w110=None):
    """Surface-weighted tensor product."""
    results = {}
    label_groups = _group_by_label(surface._labels)

    ref_vecs = {}
    for lab in label_groups:
        if w100 and len(w100) > 0 and lab in w100:
            ref_vecs[lab] = get_ref_vec(lab, w100, w110, denominator_name='w100')
        else:
            ref_vecs[lab] = np.zeros(3)

    for lab, mask in label_groups.items():
        r = _default_matrix()
        ref_vec = ref_vecs[lab]

        c1 = surface._verts[surface._faces[mask, 0]] - ref_vec
        c2 = surface._verts[surface._faces[mask, 1]] - ref_vec
        c3 = surface._verts[surface._faces[mask, 2]] - ref_vec
        area = surface._areas[mask]

        for i, j in _SYM_PAIRS:
            part_1 = c1[:, i] * c1[:, j] + c2[:, i] * c2[:, j] + c3[:, i] * c3[:, j]
            part_2 = c1[:, i] * c2[:, j] + c2[:, i] * c3[:, j] + c3[:, i] * c1[:, j]
            part_3 = c1[:, j] * c2[:, i] + c2[:, j] * c3[:, i] + c3[:, j] * c1[:, i]
            r.result[i, j] = float(np.sum(
                (1.0 / 18.0) * (part_1 + part_2 / 2.0 + part_3 / 2.0) * area
            ))

        results[lab] = r
    return results


def calculate_w220(surface, w200=None, w210=None):
    """Curvature-weighted tensor product via dihedral angles."""
    results = {}
    label_groups = _group_by_label(surface._labels)

    ref_vecs = {}
    for lab in label_groups:
        if w200 and len(w200) > 0 and lab in w200:
            ref_vecs[lab] = get_ref_vec(lab, w200, w210, denominator_name='w200')
        else:
            ref_vecs[lab] = np.zeros(3)

    # Precompute per-edge contributions for all triangles
    F = surface.n_triangles()
    # For each edge j, accumulate alpha * e * outer_product_terms
    # We compute all 6 symmetric entries at once per edge

    for lab, mask in label_groups.items():
        r = _default_matrix()
        ref_vec = ref_vecs[lab]

        tri_indices = np.where(mask)[0]
        for j in range(3):
            j1 = (j + 1) % 3
            alpha = surface._dihedral_angles[tri_indices, j]
            e_len = surface._edge_lengths[tri_indices, j]
            e_c1 = surface._verts[surface._faces[tri_indices, j]] - ref_vec
            e_c2 = surface._verts[surface._faces[tri_indices, j1]] - ref_vec
            coeff = alpha * e_len / 36.0  # 1/(18*2) = 1/36

            for i_idx, j_idx in _SYM_PAIRS:
                val = coeff * (
                    e_c1[:, i_idx] * e_c1[:, j_idx]
                    + 0.5 * (e_c1[:, i_idx] * e_c2[:, j_idx] + e_c1[:, j_idx] * e_c2[:, i_idx])
                    + e_c2[:, i_idx] * e_c2[:, j_idx]
                )
                r.result[i_idx, j_idx] += float(np.sum(val))

        results[lab] = r
    return results


def calculate_w320(surface, w300=None, w310=None):
    """Gaussian curvature-weighted tensor product via angle deficit."""
    if np.any(surface._vertex_angle_sums <= 0):
        warnings.warn(
            "Mesh contains vertices with zero or negative angle sum (isolated or "
            "degenerate triangles); their contribution is set to zero.",
            stacklevel=2,
        )

    results = {}
    label_groups = _group_by_label(surface._labels)

    ref_vecs = {}
    for lab in label_groups:
        if w300 and len(w300) > 0 and lab in w300:
            ref_vecs[lab] = get_ref_vec(lab, w300, w310, denominator_name='w300')
        else:
            ref_vecs[lab] = np.zeros(3)

    for lab, mask in label_groups.items():
        r = _default_matrix()
        ref_vec = ref_vecs[lab]

        tri_indices = np.where(mask)[0]
        for j in range(3):
            vert_idx = surface._faces[tri_indices, j]
            angle = surface._vertex_angles[tri_indices, j]
            angle_sum = surface._vertex_angle_sums[vert_idx]
            safe_angle_sum = np.where(angle_sum > 0, angle_sum, 1.0)
            angle_part = np.where(
                angle_sum > 0,
                (2.0 * np.pi * (angle / safe_angle_sum) - angle) / 3.0,
                0.0,
            )
            c = surface._verts[vert_idx] - ref_vec

            for i_idx, j_idx in _SYM_PAIRS:
                r.result[i_idx, j_idx] += float(np.sum(
                    angle_part * c[:, i_idx] * c[:, j_idx]
                ))

        results[lab] = r
    return results


def calculate_w102(surface):
    """Normal tensor: (1/3) * sum(area * n_i * n_j)."""
    results = {}
    n = surface._normals   # (F, 3)
    a = surface._areas     # (F,)
    wa = a / 3.0           # (F,)

    for lab, mask in _group_by_label(surface._labels).items():
        r = _default_matrix()
        nm = n[mask]
        wam = wa[mask]
        for i, j in _SYM_PAIRS:
            r.result[i, j] = float(np.sum(wam * nm[:, i] * nm[:, j]))
        results[lab] = r
    return results


def calculate_w202(surface):
    """Curvature-weighted normal tensor."""
    results = {}
    label_groups = _group_by_label(surface._labels)

    for lab, mask in label_groups.items():
        r = _default_matrix()
        tri_indices = np.where(mask)[0]

        for j in range(3):
            j1 = (j + 1) % 3
            alpha = surface._dihedral_angles[tri_indices, j]
            nonzero = alpha != 0.0
            if not np.any(nonzero):
                continue
            idx = tri_indices[nonzero]
            alpha_nz = alpha[nonzero]

            n1 = surface._normals[idx]
            nb_idx = surface._neighbours[idx, j]
            n2 = surface._normals[nb_idx]

            e_len = surface._edge_lengths[idx, j]
            e_c1 = surface._verts[surface._faces[idx, j]]
            e_c2 = surface._verts[surface._faces[idx, j1]]

            # Edge direction
            e_vec_raw = e_c2 - e_c1
            safe_e_len = np.where(e_len > 0, e_len, 1.0)
            e_vec = e_vec_raw / safe_e_len[:, None]

            # Average normal direction
            n_sum = n1 + n2
            n_sum_norm = np.linalg.norm(n_sum, axis=1)
            safe_n_sum_norm = np.where(n_sum_norm > 0, n_sum_norm, 1.0)
            n_a = n_sum / safe_n_sum_norm[:, None]
            n_i = np.cross(e_vec, n_a)

            sin_alpha = np.sin(alpha_nz)
            coeff_a = e_len * (alpha_nz + sin_alpha) / 24.0
            coeff_i = e_len * (alpha_nz - sin_alpha) / 24.0

            for i_idx, j_idx in _SYM_PAIRS:
                val = (coeff_a * n_a[:, i_idx] * n_a[:, j_idx]
                       + coeff_i * n_i[:, i_idx] * n_i[:, j_idx])
                r.result[i_idx, j_idx] += float(np.sum(val))

        results[lab] = r
    return results


# ============================================================================
# Higher-rank tensor functionals
# ============================================================================

def calculate_w103(surface):
    """Rank-3 normal tensor: (1/3) * sum(area * n_i * n_j * n_k)."""
    results = {}
    n = surface._normals
    wa = surface._areas / 3.0

    for lab, mask in _group_by_label(surface._labels).items():
        r = _default_rank3()
        nm = n[mask]
        wam = wa[mask]
        # Compute (F, 3, 3, 3) would be wasteful; use 10 unique entries for sym tensor
        # But Rank3Tensor stores full 27, so just do triple loop vectorized
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    r.result[i, j, k] = float(np.sum(
                        wam * nm[:, i] * nm[:, j] * nm[:, k]
                    ))
        results[lab] = r
    return results


def calculate_w104(surface):
    """Rank-4 normal tensor via fourth tensorial power."""
    results = {}
    n = surface._normals
    wa = surface._areas / 3.0
    sqrt2 = np.sqrt(2.0)

    for lab, mask in _group_by_label(surface._labels).items():
        r = _default_rank4()
        nm = n[mask]     # (N, 3)
        wam = wa[mask]   # (N,)

        # Voigt vector: [xx, yy, zz, yz*sqrt2, xz*sqrt2, xy*sqrt2]
        t = np.empty((len(wam), 6), dtype=np.float64)
        t[:, 0] = nm[:, 0] * nm[:, 0]
        t[:, 1] = nm[:, 1] * nm[:, 1]
        t[:, 2] = nm[:, 2] * nm[:, 2]
        t[:, 3] = nm[:, 1] * nm[:, 2] * sqrt2
        t[:, 4] = nm[:, 0] * nm[:, 2] * sqrt2
        t[:, 5] = nm[:, 0] * nm[:, 1] * sqrt2

        # Outer product weighted sum: sum(w * t_i * t_j)
        for i in range(6):
            for j in range(i + 1):
                r.result[i, j] = float(np.sum(wam * t[:, i] * t[:, j]))

        results[lab] = r
    return results
