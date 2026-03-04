"""
High-level API for computing Minkowski functionals from numpy arrays.
"""

import warnings

import numpy as np

from .triangulation import Triangulation
from .minkowski import (
    calculate_w000, calculate_w100, calculate_w200, calculate_w300,
    calculate_w010, calculate_w110, calculate_w210, calculate_w310,
    calculate_w020, calculate_w120, calculate_w220, calculate_w320,
    calculate_w102, calculate_w202, calculate_w103, calculate_w104,
)
from .spherical import calculate_sphmink
from .eigensystem import calculate_eigensystem

# The 14 "standard" functionals
_STANDARD = {
    'w000', 'w100', 'w200', 'w300',
    'w010', 'w110', 'w210', 'w310',
    'w020', 'w120', 'w220', 'w320',
    'w102', 'w202',
}

# Extra functionals available with compute='all'
_EXTRA = {'w103', 'w104', 'msm'}

# Rank-2 tensor names that get eigensystems
_RANK2 = ['w020', 'w120', 'w220', 'w320', 'w102', 'w202']

# Derived-quantity dependency map: derived_key → tuple of required parent keys
# Requesting a derived key auto-promotes its parents into `wanted`.
_DERIVED_DEPS = {
    # beta = min(|λ|) / max(|λ|) for each rank-2 tensor
    'w020_beta': ('w020',), 'w120_beta': ('w120',),
    'w220_beta': ('w220',), 'w320_beta': ('w320',),
    'w102_beta': ('w102',), 'w202_beta': ('w202',),
    # trace = Tr(wX20) for the wX20 family and Tr(w102), Tr(w202)
    'w020_trace': ('w020',), 'w120_trace': ('w120',),
    'w220_trace': ('w220',), 'w320_trace': ('w320',),
    'w102_trace': ('w102',), 'w202_trace': ('w202',),
    # trace_ratio = Tr(wX20) / wX00 (wX20 family only; w102/w202 have no scalar pair)
    'w020_trace_ratio': ('w020', 'w000'), 'w120_trace_ratio': ('w120', 'w100'),
    'w220_trace_ratio': ('w220', 'w200'), 'w320_trace_ratio': ('w320', 'w300'),
}

_ALL = _STANDARD | _EXTRA | set(_DERIVED_DEPS.keys())


def _extract_result(mink_result):
    """Convert a MinkValResult to a plain numpy value."""
    r = mink_result.result
    if isinstance(r, (int, float)):
        return float(r)
    if isinstance(r, np.ndarray):
        return r.copy()
    if hasattr(r, 'to_numpy'):
        return r.to_numpy()
    return r


def _build_label_dict(raw, wanted, label):
    """Extract the result for a single label from a raw dict."""
    if label in raw:
        return _extract_result(raw[label])
    return None


def minkowski_functionals(verts, faces, labels=None, center=None, compute='standard',
                          compute_eigensystems=True):
    """Compute Minkowski functionals on a triangulated surface.

    Parameters
    ----------
    verts : (V, 3) array_like
        Vertex positions.
    faces : (F, 3) array_like
        Triangle vertex indices.
    labels : (F,) array_like or None
        Per-face labels. If None, treat as a single body.
    center : None, 'centroid', or (3,) array_like
        Reference point for position-dependent tensors.
        None: use origin. 'centroid': use per-functional centroid.
        (3,) array: shift vertices by -center before computing.
    compute : str or list of str
        'standard' (14 base functionals + eigensystems),
        'all' (adds w103, w104, msm), or list of names.
    compute_eigensystems : bool, optional
        Whether to compute eigenvalues and eigenvectors for each rank-2
        tensor in the output. Default is True. When False, rank-2 tensors
        (3×3 matrices) are still computed and returned; only the
        eigendecomposition (``*_eigvals`` / ``*_eigvecs`` keys) is skipped.
        This avoids six ``np.linalg.eigh`` calls per label, which can
        dominate runtime for large batch jobs that do not need eigensystems.
        Raises ``ValueError`` if any beta-derived quantity is requested
        alongside ``compute_eigensystems=False``.

    Returns
    -------
    dict or dict[int, dict]
        When ``labels`` is ``None``: a **flat** dict mapping functional names
        to values (e.g. ``result['w000']``).

        When ``labels`` is provided: a **nested** dict keyed by label value
        (e.g. ``result[0]['w000']``).  This holds even when the labels array
        contains only a single unique value.

    Notes
    -----
    The two return shapes are asymmetric by design: the flat form is a
    convenience for the common single-body use-case.  Callers that may
    receive ``labels`` from external code should always handle the nested
    form.  A future release may deprecate the flat-dict form in favour of
    always returning ``{label: dict}``; see issue #48.
    """
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)

    # Determine which functionals to compute
    if isinstance(compute, str):
        if compute == 'standard':
            wanted = set(_STANDARD)
        elif compute == 'all':
            wanted = set(_ALL)
        else:
            raise ValueError(f"Unknown compute preset: {compute!r}")
    else:
        wanted = set(compute)
        unknown = wanted - _ALL
        if unknown:
            raise ValueError(
                f"Unknown compute names: {sorted(unknown)}. "
                f"Valid names: {sorted(_ALL)}"
            )

    # Guard: beta quantities require eigensystems
    beta_keys = {name for name in wanted if name.endswith('_beta')}
    if beta_keys and not compute_eigensystems:
        raise ValueError(
            f"{sorted(beta_keys)} requires eigensystems; "
            "set compute_eigensystems=True or remove these from compute."
        )

    # Expand derived-quantity dependencies (auto-promote parent tensors/scalars)
    for derived, parents in _DERIVED_DEPS.items():
        if derived in wanted:
            wanted.update(parents)

    # Handle explicit center by shifting vertices
    use_centroid = False
    if center is not None:
        if isinstance(center, str) and center == 'centroid':
            use_centroid = True
        else:
            center = np.asarray(center, dtype=np.float64)
            if center.shape != (3,):
                raise ValueError(
                    f"center must be a (3,) array, got shape {center.shape}"
                )
            verts = verts - center

    # Build triangulation
    face_labels = None
    if labels is not None:
        face_labels = np.asarray(labels)
    surface = Triangulation.from_arrays(verts, faces, labels=face_labels)

    # Collect all unique labels
    if labels is not None:
        unique_labels = sorted(set(int(x) for x in labels))
    else:
        unique_labels = [0]

    # --- Compute scalars ---
    # Only compute prerequisite scalars for centroid mode or direct request (#47)
    raw_w000 = calculate_w000(surface) if (
        _any_needed(wanted, ['w000', 'w010']) or (use_centroid and 'w020' in wanted)
    ) else {}
    raw_w100 = calculate_w100(surface) if (
        _any_needed(wanted, ['w100', 'w110']) or (use_centroid and 'w120' in wanted)
    ) else {}
    raw_w200 = calculate_w200(surface) if (
        _any_needed(wanted, ['w200', 'w210']) or (use_centroid and 'w220' in wanted)
    ) else {}
    raw_w300 = calculate_w300(surface) if (
        _any_needed(wanted, ['w300', 'w310']) or (use_centroid and 'w320' in wanted)
    ) else {}

    # --- Compute vectors ---
    raw_w010 = calculate_w010(surface) if (
        'w010' in wanted or (use_centroid and 'w020' in wanted)
    ) else {}
    raw_w110 = calculate_w110(surface) if (
        'w110' in wanted or (use_centroid and 'w120' in wanted)
    ) else {}
    raw_w210 = calculate_w210(surface) if (
        'w210' in wanted or (use_centroid and 'w220' in wanted)
    ) else {}
    raw_w310 = calculate_w310(surface) if (
        'w310' in wanted or (use_centroid and 'w320' in wanted)
    ) else {}

    # --- Compute rank-2 tensors ---
    if 'w020' in wanted:
        if use_centroid:
            raw_w020 = calculate_w020(surface, raw_w000, raw_w010)
        else:
            raw_w020 = calculate_w020(surface)
    else:
        raw_w020 = {}

    if 'w120' in wanted:
        if use_centroid:
            raw_w120 = calculate_w120(surface, raw_w100, raw_w110)
        else:
            raw_w120 = calculate_w120(surface)
    else:
        raw_w120 = {}

    if 'w220' in wanted:
        if use_centroid:
            raw_w220 = calculate_w220(surface, raw_w200, raw_w210)
        else:
            raw_w220 = calculate_w220(surface)
    else:
        raw_w220 = {}

    if 'w320' in wanted:
        if use_centroid:
            raw_w320 = calculate_w320(surface, raw_w300, raw_w310)
        else:
            raw_w320 = calculate_w320(surface)
    else:
        raw_w320 = {}

    raw_w102 = calculate_w102(surface) if 'w102' in wanted else {}
    raw_w202 = calculate_w202(surface) if 'w202' in wanted else {}

    # --- Optional higher-order ---
    raw_w103 = calculate_w103(surface) if 'w103' in wanted else {}
    raw_w104 = calculate_w104(surface) if 'w104' in wanted else {}
    raw_msm = calculate_sphmink(surface) if 'msm' in wanted else {}

    # Map names to raw results
    all_raw = {
        'w000': raw_w000, 'w100': raw_w100, 'w200': raw_w200, 'w300': raw_w300,
        'w010': raw_w010, 'w110': raw_w110, 'w210': raw_w210, 'w310': raw_w310,
        'w020': raw_w020, 'w120': raw_w120, 'w220': raw_w220, 'w320': raw_w320,
        'w102': raw_w102, 'w202': raw_w202,
        'w103': raw_w103, 'w104': raw_w104, 'msm': raw_msm,
    }

    # Eigensystem raw results
    eig_raw = {}
    if compute_eigensystems:
        for name in _RANK2:
            if name in wanted and all_raw[name]:
                eig_raw[name] = calculate_eigensystem(all_raw[name])

    # Build per-label output dicts
    per_label = {}
    for label in unique_labels:
        out = {}
        for name in wanted:
            raw = all_raw.get(name)
            if raw is None or label not in raw:
                continue
            if name == 'msm':
                sph = raw[label].result
                out['msm_ql'] = np.array(sph.ql)
                out['msm_wl'] = np.array(sph.wl)
            else:
                out[name] = _extract_result(raw[label])

        # Add eigensystems
        for name in _RANK2:
            if name in wanted and name in eig_raw and label in eig_raw[name]:
                es = eig_raw[name][label].result
                out[f'{name}_eigvals'] = np.array(es.eigen_values)
                out[f'{name}_eigvecs'] = np.array(es.eigen_vectors).T  # columns = eigenvectors

        # Beta (anisotropy scalar): min(|λ|) / max(|λ|), NaN when max|λ|≈0
        for name in _RANK2:
            beta_key = f'{name}_beta'
            if beta_key in wanted and f'{name}_eigvals' in out:
                abs_eigs = np.abs(out[f'{name}_eigvals'])
                max_eig = float(abs_eigs.max())
                if max_eig < 1e-12:
                    warnings.warn(
                        f"Beta for {name} (label {label}): max|eigenvalue| is near zero; "
                        "returning NaN.",
                        stacklevel=2,
                    )
                    out[beta_key] = float('nan')
                else:
                    out[beta_key] = float(abs_eigs.min()) / max_eig

        # Traces and trace ratios
        _TRACE_DENOM = {
            'w020': 'w000', 'w120': 'w100',
            'w220': 'w200', 'w320': 'w300',
        }
        for tensor_name in _RANK2:
            trace_key = f'{tensor_name}_trace'
            ratio_key = f'{tensor_name}_trace_ratio'
            needs_trace = trace_key in wanted
            needs_ratio = ratio_key in wanted
            if not (needs_trace or needs_ratio):
                continue
            if tensor_name not in all_raw or label not in all_raw[tensor_name]:
                continue
            mat = all_raw[tensor_name][label].result.to_numpy()
            trace = float(np.trace(mat))
            if needs_trace:
                out[trace_key] = trace
            if needs_ratio:
                scalar_name = _TRACE_DENOM.get(tensor_name)
                if scalar_name is not None and scalar_name in out:
                    denom = out[scalar_name]
                    out[ratio_key] = float('nan') if abs(denom) < 1e-12 else trace / denom

        per_label[label] = out

    # Warn if any label has negative volume (likely inverted face winding)
    if 'w000' in wanted:
        for lab, out in per_label.items():
            if 'w000' in out and out['w000'] < 0:
                warnings.warn(
                    f"Negative volume (w000={out['w000']:.6g}) detected"
                    f"{f' for label {lab}' if labels is not None else ''}. "
                    "This usually indicates inverted face winding (inward normals).",
                    stacklevel=2,
                )

    # Return flat dict for single-body case
    if labels is None:
        return per_label[0]
    return per_label


def _any_needed(wanted, names):
    """Check if any of the given names are in the wanted set."""
    return bool(wanted.intersection(names))


def minkowski_functionals_from_label_image(
    label_image, level=None, spacing=(1.0, 1.0, 1.0),
    center='centroid_mesh', compute='standard', compute_eigensystems=True
):
    """Compute Minkowski functionals for each label in a 3D label image.

    Parameters
    ----------
    label_image : (Z, Y, X) array_like of int
        3D label image where each unique nonzero value identifies an object.
    level : float or None
        Isosurface level for marching_cubes. Default ``0.5`` (suitable for
        binary masks).
    spacing : tuple of float
        ``(sz, sy, sx)`` voxel spacing passed to ``marching_cubes``.
    center : None, 'centroid_mesh', 'centroid_voxel', or (3,) array_like
        Reference point for position-dependent tensors.
        ``'centroid_mesh'`` (default): centroid of the volume enclosed by the mesh,
            computed via the divergence theorem (each triangle and the origin form a
            tetrahedron; the centroid is the volume-weighted mean of tetrahedron
            centroids). Requires outward-facing normals.
        ``'centroid_voxel'``: centroid of the set of labelled voxels (mean of
            voxel-grid coordinates scaled by spacing), consistent with
            scikit-image ``regionprops`` convention.
        ``None``: use the origin.
        ``(3,)`` array: use an explicit point for all labels.
    compute : str or list of str
        Which functionals to compute. ``'standard'`` returns the 14 base
        functionals; ``'all'`` additionally computes ``w103``, ``w104``,
        and spherical Minkowski summaries (``msm_ql``, ``msm_wl``); a list
        of names selects specific quantities.
    compute_eigensystems : bool, optional
        If False, eigenvalues and eigenvectors for rank-2 tensors are
        skipped (``*_eigvals`` / ``*_eigvecs`` keys are omitted from each
        label's result dict), avoiding six ``np.linalg.eigh`` calls per
        label. Default is True.

    Returns
    -------
    dict[int, dict]
        Mapping from label value to a dict of Minkowski functionals.

    Notes
    -----
    Requires *scikit-image* (``skimage.measure.marching_cubes``).
    """
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        raise ImportError(
            "scikit-image is required for minkowski_functionals_from_label_image. "
            "Install it with: pip install scikit-image"
        )

    label_image = np.asarray(label_image)
    if not np.issubdtype(label_image.dtype, np.integer):
        warnings.warn(
            f"label_image has dtype {label_image.dtype}; converting to int. "
            "Pass an integer-dtype array to suppress this warning.",
            stacklevel=2,
        )
        label_image = label_image.astype(int)
    if level is None:
        level = 0.5
    spacing = tuple(float(s) for s in spacing)

    unique_labels = np.unique(label_image)
    unique_labels = unique_labels[unique_labels != 0]

    results = {}
    for lab in unique_labels:
        lab = int(lab)
        mask = (label_image == lab).astype(np.float64)

        try:
            verts, faces, _, _ = marching_cubes(mask, level=level, spacing=spacing,
                                               gradient_direction='ascent')
        except Exception as exc:
            warnings.warn(
                f"marching_cubes failed for label {lab}: {exc}",
                stacklevel=2,
            )
            continue

        # marching_cubes may produce inward-facing normals; ensure outward
        # by checking the signed volume and flipping faces if negative.
        v0 = verts[faces[:, 0]]
        cross = np.cross(verts[faces[:, 1]] - v0, verts[faces[:, 2]] - v0)
        signed_vol = np.sum(v0 * cross) / 6.0
        if signed_vol < 0:
            faces = faces[:, ::-1]

        # Determine center for this label
        if isinstance(center, str) and center == 'centroid_mesh':
            # Volume-weighted centroid via the divergence theorem.
            # Runs after the face-flip so normals are guaranteed outward.
            v0c = verts[faces[:, 0]]
            v1c = verts[faces[:, 1]]
            v2c = verts[faces[:, 2]]
            d = np.einsum('ij,ij->i', v0c, np.cross(v1c - v0c, v2c - v0c))
            d_sum = d.sum()
            if abs(d_sum) < 1e-12:
                warnings.warn(
                    f"Mesh centroid denominator near zero for label {lab}; "
                    "falling back to origin reference.",
                    stacklevel=2,
                )
                label_center = np.zeros(3)
            else:
                label_center = np.einsum('i,ij->j', d, v0c + v1c + v2c) / (4.0 * d_sum)
        elif isinstance(center, str) and center == 'centroid_voxel':
            voxel_coords = np.argwhere(label_image == lab)  # (N, 3)
            label_center = voxel_coords.mean(axis=0) * np.array(spacing)
        else:
            label_center = center

        results[lab] = minkowski_functionals(
            verts, faces, center=label_center, compute=compute,
            compute_eigensystems=compute_eigensystems,
        )

    return results
