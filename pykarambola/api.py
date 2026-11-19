# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2026 Keisuke Ishihara, Yajushi Khurana
# This file is part of pykarambola, a Python port of karambola.
# See LICENSE for the full license text.
"""
High-level API for computing Minkowski tensors from numpy arrays.
"""

import warnings
from typing import Union

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as _sp_connected_components

from .triangulation import Triangulation, LABEL_UNASSIGNED, NEIGHBOUR_UNASSIGNED
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

# Denominator scalar for each wX20 trace ratio (wX20 family only).
# This intentionally duplicates information already in _DERIVED_DEPS; do not
# consolidate — _DERIVED_DEPS drives dependency promotion while _TRACE_DENOM
# drives the runtime scalar lookup inside the per-label compute loop.
_TRACE_DENOM = {
    'w020': 'w000', 'w120': 'w100',
    'w220': 'w200', 'w320': 'w300',
}


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


def _compute_mesh_centroid(verts, faces):
    """Volume-weighted centroid via the divergence theorem (equivalent to w010/w000).

    Returns ``(centroid, success)``. ``success`` is False when the denominator
    is near zero; the caller should fall back to the origin in that case.
    """
    if len(faces) == 0:
        return np.zeros(3), False
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    d = np.einsum('ij,ij->i', v0, np.cross(v1 - v0, v2 - v0))
    d_sum = d.sum()
    if abs(d_sum) < 1e-12:
        return np.zeros(3), False
    return np.einsum('i,ij->j', d, v0 + v1 + v2) / (4.0 * d_sum), True


def _ensure_outward_normals(verts, faces):
    """Return faces with outward-facing normals, flipping winding if needed."""
    v0 = verts[faces[:, 0]]
    cross = np.cross(verts[faces[:, 1]] - v0, verts[faces[:, 2]] - v0)
    if np.sum(v0 * cross) / 6.0 < 0:
        return faces[:, ::-1]
    return faces


def _label_mesh_components(faces):
    """Return a per-face integer label (1-based) for each connected component.

    Parameters
    ----------
    faces : (F, 3) array_like
        Triangle vertex indices (any non-negative integer values).

    Returns
    -------
    labels : np.ndarray, shape (F,)
        Component label for each face, integers starting from 1.
        Empty array when ``faces`` is empty.
    """
    faces = np.asarray(faces, dtype=np.int64)
    if len(faces) == 0:
        return np.empty(0, dtype=np.intp)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]])
    n = int(faces.max()) + 1
    data = np.ones(len(edges), dtype=np.int8)
    graph = csr_matrix((data, (edges[:, 0], edges[:, 1])), shape=(n, n))
    _, vertex_labels = _sp_connected_components(graph, directed=False)
    raw = vertex_labels[faces[:, 0]]
    _, contiguous = np.unique(raw, return_inverse=True)
    return contiguous.astype(np.intp) + 1


def _count_mesh_components(faces):
    """Return the number of disconnected components in a triangular mesh.

    Parameters
    ----------
    faces : (F, 3) array_like
        Triangle vertex indices (any non-negative integer values).

    Returns
    -------
    int
    """
    component_labels = _label_mesh_components(np.asarray(faces))
    if len(component_labels) == 0:
        return 0
    return int(np.unique(component_labels).size)


def _get_open_and_nonmanifold(surface):
    """Return (open_labels, is_non_manifold) for a finalized Triangulation.

    open_labels : set of int
        Labels that have at least one boundary (open) edge.
    is_non_manifold : bool
        True if any vertex is non-manifold (more objects meet at a single vertex
        than a closed or open fan allows).
    """
    nb = surface._neighbours   # (F, 3)
    labels = surface._labels   # (F,)

    # Open-edge detection per label
    has_open = np.any(nb == NEIGHBOUR_UNASSIGNED, axis=1)
    open_labels = (
        set(int(l) for l in np.unique(labels[has_open]))
        if np.any(has_open)
        else set()
    )

    # Non-manifold detection via fan traversal.
    #
    # Precompute in one O(F) NumPy pass: for every CSR entry (vertex v,
    # triangle t), the "fan-next" triangle obtained by crossing the edge of t
    # that starts at v (i.e. nb[t, slot_of_v_in_t]).  The per-vertex loop then
    # uses a tiny dict (d ≤ ~12 entries) instead of Python method calls and a
    # generator expression, eliminating ~5 Python function calls per fan step.
    V = surface.n_vertices()
    F = surface.n_triangles()
    if V == 0 or F == 0:
        return open_labels, False

    faces_arr = surface._faces  # (F, 3) int64

    if surface._vt_offsets is not None:
        vt_off = surface._vt_offsets    # (V+1,) int64
        vt_idx = surface._vt_indices    # (3F,)  int64
        # vertex that owns each CSR entry
        deg = np.diff(vt_off).astype(np.int64)
        entry_vert = np.repeat(np.arange(V, dtype=np.int64), deg)   # (3F,)
        # slot of that vertex in the corresponding triangle
        _match = faces_arr[vt_idx] == entry_vert[:, None]           # (3F, 3)
        if not np.all(np.any(_match, axis=1)):
            raise AssertionError("CSR integrity error: vertex not found in its own triangle.")
        entry_slot = np.argmax(_match, axis=1)                      # (3F,)
        # fan-next triangle for each CSR entry
        entry_succ = nb[vt_idx, entry_slot]                          # (3F,)
        use_csr = True
    else:
        vt_off = vt_idx = entry_succ = None
        use_csr = False

    is_non_manifold = False
    for i in range(V):
        if use_csr:
            s = int(vt_off[i])
            e = int(vt_off[i + 1])
            d = e - s
            if d <= 1:
                continue
            tris_i = vt_idx[s:e]       # (d,) triangle indices
            succs_i = entry_succ[s:e]  # (d,) fan-next triangles
        else:
            tris_list = surface._vertex_triangles[i]
            d = len(tris_list)
            if d <= 1:
                continue
            tris_i = np.asarray(tris_list, dtype=np.int64)
            slots_i = np.argmax(faces_arr[tris_i] == i, axis=1)
            succs_i = nb[tris_i, slots_i]

        # Build triangle-id → local-index map (d ≤ ~12, so dict overhead is negligible)
        local = {int(tris_i[j]): j for j in range(d)}

        start_t = int(tris_i[0])
        cur_t = start_t
        sum_tris = 1
        neigh_un = False

        for _ in range(d):
            nxt = int(succs_i[local[cur_t]])
            if nxt == NEIGHBOUR_UNASSIGNED:
                neigh_un = True
                break
            if nxt == start_t:
                break
            sum_tris += 1
            cur_t = nxt

        if not neigh_un and d != sum_tris:
            is_non_manifold = True
            break

    return open_labels, is_non_manifold


def minkowski_tensors(verts: Union[np.ndarray, Triangulation], faces=None, labels=None, center=None, center_per_label=True,
                      compute='standard', compute_eigensystems=True, return_count=False, msm_max_l=8):
    """Compute Minkowski tensors on a triangulated surface.

    Parameters
    ----------
    verts : (V, 3) array_like or Triangulation
        Vertex positions, or a ``Triangulation`` object returned by any parser
        (``parse_obj_file``, ``parse_off_file``, ``parse_poly_file``,
        ``parse_glb_file``). When a ``Triangulation`` is passed, ``faces``
        may be omitted and ``labels`` defaults to the triangulation's own
        label array (if any).
    faces : (F, 3) array_like or None
        Triangle vertex indices. Required when *verts* is an array; ignored
        (may be ``None``) when *verts* is a ``Triangulation``.
    labels : (F,) array_like, 'auto', or None
        Per-face labels. If None, treat as a single body.
        If ``'auto'``, connected mesh components are detected automatically
        and assigned 1-based integer labels (``1``, ``2``, …). The return
        value has the same nested-dict structure as the explicit-labels case.
    center : None, 'reference_centroid', 'centroid_mesh', or (3,) array_like
        Reference point for position-dependent tensors.

        ``None``: use the origin (0, 0, 0).

        ``'reference_centroid'``: per-tensor Minkowski centroid, equivalent
            to the C++ ``--reference_centroid`` flag. Each rank-2 tensor uses
            its own reference vector derived from its prerequisite Minkowski
            scalar and vector (e.g. ``w020`` uses ``w010 / w000``). Always
            computed per-label regardless of ``center_scope``.

        ``'centroid_mesh'``: volume-weighted center of mass computed via the
            divergence theorem (``w010 / w000`` geometrically). Vertices are
            shifted by ``-centroid`` before all computations. See
            ``center_per_label`` for per-label vs global behavior when
            ``labels`` is provided.

        ``(3,)`` array: shift all vertices by ``-center`` before computing.

        .. deprecated::
            ``center='centroid'`` is deprecated; use
            ``center='reference_centroid'`` instead.

    center_per_label : bool, optional
        Controls centroid scope when ``labels`` is provided and
        ``center='centroid_mesh'``. Ignored for ``'reference_centroid'``
        (always per-label) and explicit array centers.

        ``True`` (default): centroid computed independently for each
            labeled sub-mesh. Matches the behavior of
            ``minkowski_tensors_from_label_image``.

        ``False``: centroid computed from the entire mesh (all faces)
            and applied as a single shift to every label.

    compute : str or list of str
        'standard' (14 base tensors + eigensystems),
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
    return_count : bool, optional
        If True, return a ``(results, n_objects)`` tuple where ``n_objects``
        is the total number of connected components across all labels,
        determined by vertex-adjacency graph traversal. Default is False.
    msm_max_l : int, optional
        Maximum spherical harmonic order for spherical Minkowski tensors
        (``msm``). Default is 8, matching C++ karambola. Output arrays
        ``msm_ql`` and ``msm_wl`` have shape ``(msm_max_l + 1,)``.

    Returns
    -------
    dict or dict[int, dict]
        When ``labels`` is ``None``: a **flat** dict mapping tensor names
        to values (e.g. ``result['w000']``).

        When ``labels`` is provided: a **nested** dict keyed by label value
        (e.g. ``result[0]['w000']``).  This holds even when the labels array
        contains only a single unique value.

    tuple (dict or dict[int, dict], int)
        When ``return_count=True``: ``(results, n_objects)``.

    Notes
    -----
    The two return shapes are asymmetric by design: the flat form is a
    convenience for the common single-body use-case.  Callers that may
    receive ``labels`` from external code should always handle the nested
    form.  A future release may deprecate the flat-dict form in favour of
    always returning ``{label: dict}``; see issue #48.
    """
    # Issue #90: accept Triangulation objects directly
    if isinstance(verts, Triangulation):
        tri = verts
        tri._consolidate_lists()
        if faces is None:
            faces = tri._faces
        if labels is None and tri._labels is not None:
            raw_labels = tri._labels
            unique_raw = np.unique(raw_labels)
            # Use the triangulation's labels only when they carry meaningful
            # per-body information: skip if all faces share a single default
            # label (0 from from_arrays, or LABEL_UNASSIGNED from parsers).
            has_meaningful_labels = not (
                len(unique_raw) == 1 and unique_raw[0] in (0, LABEL_UNASSIGNED)
            )
            if has_meaningful_labels:
                labels = raw_labels
        verts = tri._verts
    elif faces is None:
        raise TypeError(
            "faces must be provided when verts is not a Triangulation"
        )

    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)

    # Expand 'auto' labels: detect connected mesh components
    if isinstance(labels, str) and labels == 'auto':
        labels = _label_mesh_components(faces)

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

    if msm_max_l < 0:
        raise ValueError(f"msm_max_l must be >= 0, got {msm_max_l}")

    # Guard: beta quantities require eigensystems
    beta_keys = {name for name in wanted if name.endswith('_beta')}
    if beta_keys and not compute_eigensystems:
        if isinstance(compute, str):
            raise ValueError(
                f"{sorted(beta_keys)} requires eigensystems. "
                "Either set compute_eigensystems=True, or pass a list of names "
                "that excludes beta quantities."
            )
        raise ValueError(
            f"{sorted(beta_keys)} requires eigensystems; "
            "set compute_eigensystems=True or remove these from compute."
        )

    # Expand derived-quantity dependencies (auto-promote parent tensors/scalars)
    for derived, parents in _DERIVED_DEPS.items():
        if derived in wanted:
            wanted.update(parents)

    # --- Special dispatch: centroid_mesh + per_label + labels provided ---
    # Each labeled sub-mesh gets its own center of mass shift before dispatch.
    if isinstance(center, str) and center == 'centroid_mesh' and labels is not None and center_per_label:
        face_labels_arr = np.asarray(labels, dtype=np.int64)
        per_label_out = {}
        n_objects = 0
        for lab in sorted(set(int(x) for x in face_labels_arr)):
            lab_faces_global = faces[face_labels_arr == lab]
            if len(lab_faces_global) == 0:
                continue
            used_idx = np.unique(lab_faces_global)
            sub_verts = verts[used_idx]
            remap = np.full(verts.shape[0], -1, dtype=np.int64)
            remap[used_idx] = np.arange(len(used_idx), dtype=np.int64)
            sub_faces = remap[lab_faces_global]
            centroid, ok = _compute_mesh_centroid(sub_verts, sub_faces)
            if not ok:
                warnings.warn(
                    f"Mesh centroid denominator near zero for label {lab}; "
                    "falling back to origin reference.",
                    stacklevel=2,
                )
                shifted_verts = sub_verts
            else:
                shifted_verts = sub_verts - centroid
            if return_count:
                n_objects += _count_mesh_components(sub_faces)
            per_label_out[lab] = minkowski_tensors(
                shifted_verts, sub_faces, labels=None, center=None,
                compute=compute, compute_eigensystems=compute_eigensystems,
                return_count=False, msm_max_l=msm_max_l,
            )
        if return_count:
            return per_label_out, n_objects
        return per_label_out

    # --- centroid_mesh global scope (or no labels): shift full mesh then proceed ---
    if isinstance(center, str) and center == 'centroid_mesh':
        centroid, ok = _compute_mesh_centroid(verts, faces)
        if not ok:
            warnings.warn(
                "Mesh centroid denominator near zero; falling back to origin reference.",
                stacklevel=2,
            )
        else:
            verts = verts - centroid
        center = None  # vertices already shifted; proceed with origin

    # Handle explicit center by shifting vertices
    use_centroid = False
    if center is not None:
        if isinstance(center, str) and center == 'reference_centroid':
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
        face_labels = np.asarray(labels, dtype=np.int64)
    surface = Triangulation.from_arrays(verts, faces, labels=face_labels)

    # Issue #94: warn for open / non-manifold surfaces
    _open_labels, _is_non_manifold = _get_open_and_nonmanifold(surface)
    if _open_labels:
        warnings.warn(
            "Open surface detected"
            f" (label{'s' if len(_open_labels) > 1 else ''} {sorted(_open_labels)}): "
            "volume-dependent quantities (w000, w020) are unreliable for open meshes "
            "and will be set to NaN.",
            UserWarning,
            stacklevel=2,
        )
    if _is_non_manifold:
        warnings.warn(
            "Non-manifold mesh detected: results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

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
    raw_msm = calculate_sphmink(surface, max_l=msm_max_l) if 'msm' in wanted else {}

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
                if name == 'w104':
                    out['w104_eigvals'] = np.linalg.eigh(out['w104'])[0]

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
                    if abs(denom) < 1e-12:
                        warnings.warn(
                            f"Trace ratio for {tensor_name} (label {label}): "
                            f"{scalar_name} is near zero; returning NaN.",
                            stacklevel=2,
                        )
                        out[ratio_key] = float('nan')
                    else:
                        out[ratio_key] = trace / denom

        per_label[label] = out

    # Issue #94: patch NaN for volume-dependent quantities on open surfaces
    for lab in _open_labels:
        if lab in per_label:
            if 'w000' in per_label[lab]:
                per_label[lab]['w000'] = float('nan')
            if 'w020' in per_label[lab]:
                per_label[lab]['w020'] = np.full((3, 3), np.nan)

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

    # Build final result
    result = per_label[0] if labels is None else per_label

    if return_count:
        face_labels_arr = np.asarray(labels, dtype=np.int64) if labels is not None else None
        if labels is not None:
            n_objects = sum(
                _count_mesh_components(faces[face_labels_arr == lab])
                for lab in unique_labels
            )
        else:
            n_objects = _count_mesh_components(faces)
        return result, n_objects

    return result


def _any_needed(wanted, names):
    """Check if any of the given names are in the wanted set."""
    return bool(wanted.intersection(names))


def minkowski_tensors_from_label_image(
    label_image, level=None, spacing=(1.0, 1.0, 1.0),
    center='centroid_mesh', center_per_label=True,
    compute='standard', compute_eigensystems=True, return_count=False,
    autolabel=False, pad=True, msm_max_l=8, return_meshes=False,
):
    """Compute Minkowski tensors for each label in a 3D label image.

    Parameters
    ----------
    label_image : (Z, Y, X) array_like of int
        3D label image where each unique nonzero value identifies an object.
    level : float or None
        Isosurface level for marching_cubes. Default ``0.5`` (suitable for
        binary masks).
    spacing : tuple of float
        ``(sz, sy, sx)`` voxel spacing passed to ``marching_cubes``.
    center : None, 'centroid_mesh', 'centroid_voxel', 'reference_centroid', or (3,) array_like
        Reference point for position-dependent tensors.

        ``'centroid_mesh'`` (default): volume-weighted center of mass computed
            via the divergence theorem (equivalent to ``w010 / w000``). See
            ``center_per_label`` for per-label vs global behavior.

        ``'centroid_voxel'``: centroid of the set of labelled voxels (mean of
            voxel-grid coordinates scaled by spacing), consistent with
            scikit-image ``regionprops`` convention. See ``center_per_label``.

        ``'reference_centroid'``: per-tensor Minkowski centroid, equivalent to
            the C++ ``--reference_centroid`` flag. Passed through to
            ``minkowski_tensors`` unchanged; always computed per-label.

        ``None``: use the origin (0, 0, 0) — the corner of the numpy array,
            following numpy array indexing convention.

        ``(3,)`` array: explicit point applied to all labels.

    center_per_label : bool, optional
        Controls centroid scope for ``'centroid_mesh'`` and ``'centroid_voxel'``.
        Ignored for ``'reference_centroid'`` and explicit array centers.

        ``True`` (default): centroid computed independently for each
            label's mesh or voxel set.

        ``False``: a single centroid is computed from all non-zero labels
            combined and applied uniformly to all labels.

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
    return_count : bool, optional
        If True, return a ``(results, n_objects)`` tuple where ``n_objects``
        is the total number of connected components across all labels.
        Default is False.
    autolabel : bool, optional
        If False (default), compute tensors for each unique non-zero label
        value as a single body; results are keyed by label value (int).
        If True, ignore voxel label values: treat the image as binary,
        build a single mesh from all non-zero voxels, and detect connected
        components automatically (equivalent to calling
        ``minkowski_tensors(..., labels='auto')``). Results are keyed by
        0-based component index (``0``, ``1``, …).
    pad : bool, optional
        If True (default), apply a 1-voxel zero-padding on all six faces of
        the label image before passing it to ``marching_cubes``. This ensures
        that objects touching the array boundary produce closed (non-open)
        surfaces. The resulting vertex coordinates are shifted back so that
        the output is in the original image coordinate system. Set ``pad=False``
        only if the image has already been padded, or if open surfaces at
        the boundary are intentional.
    msm_max_l : int, optional
        Maximum spherical harmonic order for spherical Minkowski tensors.
        Default is 8. See ``minkowski_tensors`` for details.
    return_meshes : bool, optional
        If True, also return a ``meshes`` dict mapping each result key
        (same keys as the results dict) to the ``(verts, faces)`` mesh
        already produced internally by ``marching_cubes`` for that label.
        Reuses the mesh computed for the Minkowski tensors — no extra
        ``marching_cubes`` call is made. Pass the result to
        ``combine_label_meshes`` to build a single mesh for visualization
        (e.g. ``napari.Viewer.add_surface``). Default is False.

    Returns
    -------
    dict[int, dict]
        When ``return_count=False`` and ``return_meshes=False``: mapping
        from label value (int), or component index (int, 1-based) when
        ``autolabel=True``, to a dict of Minkowski functionals.

    tuple
        When ``return_count`` and/or ``return_meshes`` is True, the results
        dict above is augmented, in this order:

        ``(results, n_objects)`` when only ``return_count=True``.

        ``(results, meshes)`` when only ``return_meshes=True``.

        ``(results, n_objects, meshes)`` when both are True.

        ``meshes`` is a ``dict[int, (verts, faces)]`` with the same keys as
        ``results``.

    Notes
    -----
    Requires *scikit-image*.
    """
    try:
        from skimage.measure import marching_cubes, label as sk_label, regionprops
    except ImportError:
        raise ImportError(
            "scikit-image is required for minkowski_tensors_from_label_image. "
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

    # Issue #99: pad to prevent open surfaces at image boundaries
    if pad:
        label_image = np.pad(label_image, pad_width=1, mode='constant', constant_values=0)
        _pad_offset = np.array(spacing, dtype=np.float64)
    else:
        _pad_offset = np.zeros(3, dtype=np.float64)

    unique_labels = np.unique(label_image)
    unique_labels = unique_labels[unique_labels != 0]

    # Pre-compute per-label bounding boxes in one pass so the per-label loop
    # can crop the image before calling marching_cubes instead of operating on
    # the full image each time.  This is O(image_size) once rather than
    # O(N_labels × image_size).  (#156)
    _shape = label_image.shape
    _sp = np.array(spacing, dtype=np.float64)
    if len(unique_labels) > 0:
        _label_bboxes = {p.label: p.bbox for p in regionprops(label_image)}
    else:
        _label_bboxes = {}

    def _crop_label(lab_int):
        """Return (cropped_float64_mask, crop_origin_physical).

        crop_origin_physical is added to marching_cubes vertices (which are in
        crop-local space) to restore full-image coordinates before subtracting
        _pad_offset.
        """
        if lab_int in _label_bboxes:
            z0, y0, x0, z1, y1, x1 = _label_bboxes[lab_int]
            z0 = max(0, z0 - 1); y0 = max(0, y0 - 1); x0 = max(0, x0 - 1)
            z1 = min(_shape[0], z1 + 1)
            y1 = min(_shape[1], y1 + 1)
            x1 = min(_shape[2], x1 + 1)
        else:
            warnings.warn(
                f"Label {lab_int} not found in regionprops bounding-box cache; "
                "falling back to full-image crop (performance may degrade). "
                "This is unexpected - please file a bug report.",
                RuntimeWarning,
                stacklevel=3,
            )
            z0, y0, x0 = 0, 0, 0
            z1, y1, x1 = _shape
        crop = label_image[z0:z1, y0:y1, x0:x1]
        return (crop == lab_int).astype(np.float64), np.array([z0, y0, x0], dtype=np.float64) * _sp

    # --- autolabel=True: treat image as binary, delegate to minkowski_tensors ---
    if autolabel:
        # Crop to the global bounding box of all non-zero voxels (#173) so
        # marching_cubes doesn't run over empty padding/border regions.
        # Reuses the per-label bboxes already computed above (#156) instead
        # of a second full-image np.argwhere scan.
        if _label_bboxes:
            starts = np.array([b[:3] for b in _label_bboxes.values()])
            stops = np.array([b[3:] for b in _label_bboxes.values()])
            z0, y0, x0 = np.maximum(starts.min(axis=0) - 1, 0)
            z1, y1, x1 = np.minimum(stops.max(axis=0) + 1, _shape)
            crop = label_image[z0:z1, y0:y1, x0:x1]
            crop_origin = np.array([z0, y0, x0], dtype=np.float64) * _sp
        else:
            crop = label_image
            crop_origin = np.zeros(3, dtype=np.float64)

        binary = (crop != 0).astype(np.float64)
        try:
            verts, faces, _, _ = marching_cubes(binary, level=level, spacing=spacing,
                                                gradient_direction='ascent')
        except Exception as exc:
            raise RuntimeError(f"marching_cubes failed on binary mask: {exc}") from exc
        verts = verts + crop_origin - _pad_offset
        faces = _ensure_outward_normals(verts, faces)
        # centroid_voxel is not supported by minkowski_tensors; pre-compute it here.
        if isinstance(center, str) and center == 'centroid_voxel':
            nz_coords = np.argwhere(binary > 0)
            mt_center = (nz_coords.mean(axis=0) * np.array(spacing) + crop_origin - _pad_offset
                         if len(nz_coords) else np.zeros(3))
        else:
            mt_center = center
        autolabel_result = minkowski_tensors(
            verts, faces, labels='auto',
            center=mt_center, center_per_label=center_per_label,
            compute=compute, compute_eigensystems=compute_eigensystems,
            return_count=return_count, msm_max_l=msm_max_l,
        )
        if not return_meshes:
            return autolabel_result
        if return_count:
            results, n_objects_total = autolabel_result
        else:
            results = autolabel_result
        meshes = {}
        face_labels_arr = _label_mesh_components(faces)
        for comp in np.unique(face_labels_arr):
            comp_faces_global = faces[face_labels_arr == comp]
            used_idx = np.unique(comp_faces_global)
            remap = np.full(len(verts), -1, dtype=np.int64)
            remap[used_idx] = np.arange(len(used_idx), dtype=np.int64)
            meshes[int(comp)] = (verts[used_idx], remap[comp_faces_global])
        if return_count:
            return results, n_objects_total, meshes
        return results, meshes

    # --- Count connected components upfront if requested ---
    n_objects_total = 0
    if return_count:
        for lab in unique_labels:
            crop_mask, _ = _crop_label(int(lab))
            _, n = sk_label(crop_mask, return_num=True)
            n_objects_total += n

    # --- Compute global center upfront for center_scope='global' ---
    global_center = None
    label_meshes = None  # cache to avoid running marching_cubes twice
    if (not center_per_label
            and isinstance(center, str)
            and center in ('centroid_mesh', 'centroid_voxel')):
        if center == 'centroid_voxel':
            nz_coords = np.argwhere(label_image != 0)
            global_center = (nz_coords.mean(axis=0) * np.array(spacing) - _pad_offset
                             if len(nz_coords) > 0 else np.zeros(3))
        else:  # centroid_mesh global: merge all label meshes
            label_meshes = {}
            all_verts_list, all_faces_list, vert_offset = [], [], 0
            for lab in unique_labels:
                lab_int = int(lab)
                mask, crop_origin = _crop_label(lab_int)
                try:
                    v, f, _, _ = marching_cubes(mask, level=level, spacing=spacing,
                                                gradient_direction='ascent')
                    v = v + crop_origin - _pad_offset
                    f = _ensure_outward_normals(v, f)
                    label_meshes[lab_int] = (v, f)
                    all_verts_list.append(v)
                    all_faces_list.append(f + vert_offset)
                    vert_offset += len(v)
                except Exception as exc:
                    warnings.warn(
                        f"marching_cubes failed for label {lab_int}: {exc}",
                        stacklevel=2,
                    )
            if all_verts_list:
                merged_verts = np.vstack(all_verts_list)
                merged_faces = np.vstack(all_faces_list).astype(np.int64)
                gc, ok = _compute_mesh_centroid(merged_verts, merged_faces)
                if not ok:
                    warnings.warn(
                        "Global mesh centroid near zero; falling back to origin.",
                        stacklevel=2,
                    )
                    global_center = np.zeros(3)
                else:
                    global_center = gc
            else:
                global_center = np.zeros(3)

    meshes_out = None
    if return_meshes:
        meshes_out = dict(label_meshes) if label_meshes is not None else {}

    results = {}
    for lab in unique_labels:
        lab = int(lab)

        # Use cached mesh if available (avoids re-running marching_cubes)
        if label_meshes is not None and lab in label_meshes:
            cached = label_meshes[lab]
        else:
            cached = None

        if cached is None:
            crop_mask, crop_origin = _crop_label(lab)
        else:
            crop_mask = crop_origin = None

        for result_key, cached_mesh in [(lab, cached)]:
            if cached_mesh is not None:
                verts, faces = cached_mesh
            else:
                try:
                    verts, faces, _, _ = marching_cubes(crop_mask, level=level, spacing=spacing,
                                                        gradient_direction='ascent')
                except Exception as exc:
                    warnings.warn(
                        f"marching_cubes failed for {result_key}: {exc}",
                        stacklevel=2,
                    )
                    continue
                verts = verts + crop_origin - _pad_offset
                faces = _ensure_outward_normals(verts, faces)

            # Determine center for this entry
            if global_center is not None:
                label_center = global_center
            elif isinstance(center, str) and center == 'centroid_mesh':
                # Volume-weighted center of mass via the divergence theorem.
                # Uses _ensure_outward_normals result so normals are outward.
                centroid, ok = _compute_mesh_centroid(verts, faces)
                if not ok:
                    warnings.warn(
                        f"Mesh centroid denominator near zero for {result_key}; "
                        "falling back to origin reference.",
                        stacklevel=2,
                    )
                    label_center = np.zeros(3)
                else:
                    label_center = centroid
            elif isinstance(center, str) and center == 'centroid_voxel':
                if cached_mesh is not None:
                    # Recompute from the full image when using a cached mesh
                    # (crop_mask is not available in this branch).
                    voxel_coords = np.argwhere(label_image == lab)
                    label_center = voxel_coords.mean(axis=0) * _sp - _pad_offset
                else:
                    voxel_coords = np.argwhere(crop_mask > 0)  # (N, 3)
                    label_center = voxel_coords.mean(axis=0) * _sp + crop_origin - _pad_offset
            elif isinstance(center, str) and center == 'reference_centroid':
                label_center = 'reference_centroid'
            else:
                label_center = center

            results[result_key] = minkowski_tensors(
                verts, faces, center=label_center, compute=compute,
                compute_eigensystems=compute_eigensystems, msm_max_l=msm_max_l,
            )
            if meshes_out is not None and result_key not in meshes_out:
                meshes_out[result_key] = (verts, faces)

    if return_count and return_meshes:
        return results, n_objects_total, meshes_out
    if return_count:
        return results, n_objects_total
    if return_meshes:
        return results, meshes_out
    return results


def minkowski_tensors_from_scalar_field(
    field,
    level,
    spacing=(1.0, 1.0, 1.0),
    interior='above',
    center='centroid_mesh',
    center_per_label=True,
    compute='standard',
    compute_eigensystems=True,
    return_count=False,
    autolabel=False,
    pad=True,
    pad_value=None,
    msm_max_l=8,
    return_meshes=False,
):
    """Compute Minkowski tensors on an isosurface extracted from a 3D scalar field.

    Runs ``skimage.measure.marching_cubes`` directly on a continuous 3D field
    (such as a segmentation probability map, a signed distance function, or a
    calibrated intensity volume), preserving sub-voxel boundary interpolation
    instead of discretizing to a binary mask first.

    Parameters
    ----------
    field : (Z, Y, X) array_like
        3D real-valued numeric scalar field. Non-float arrays are converted
        to ``float64`` on the extracted bounding crop. All values must be
        finite. When ``pad=False``, every axis must have length at least 2.
    level : float
        Isosurface level at which the surface is extracted (required). For
        example, ``0.5`` for a ``[0, 1]`` probability map or ``0.0`` for a
        signed distance function (SDF).
    spacing : tuple of float, optional
        ``(sz, sy, sx)`` voxel spacing passed to ``marching_cubes``.
        Default is ``(1.0, 1.0, 1.0)``.
    interior : {'above', 'below'}, optional
        Which side of ``level`` represents the solid interior of the object(s):

        ``'above'`` (default): voxels with ``field > level`` are inside
            (appropriate for probability maps, binary masks, and foreground
            intensity volumes).

        ``'below'``: voxels with ``field < level`` are inside (appropriate
            for standard signed distance functions where negative values lie
            inside the object).

        This controls both outward face winding and the automatic exterior
        padding side when ``pad=True``.
    center : None, 'centroid_mesh', 'reference_centroid', or (3,) array_like, optional
        Reference point for position-dependent tensors.

        ``'centroid_mesh'`` (default): volume-weighted center of mass computed
            via the divergence theorem (equivalent to ``w010 / w000``). See
            ``center_per_label`` for per-component vs global behavior when
            ``autolabel=True``.

        ``'reference_centroid'``: per-tensor Minkowski centroid, equivalent to
            the C++ ``--reference_centroid`` flag.

        ``None``: use the origin ``(0, 0, 0)``.

        ``(3,)`` array: explicit reference point in physical coordinates.

        Note: ``'centroid_voxel'`` is not supported for scalar fields.
    center_per_label : bool, optional
        Controls centroid scope for ``center='centroid_mesh'`` when
        ``autolabel=True``. Ignored when ``autolabel=False``, or for
        ``'reference_centroid'`` and explicit array centers. Default is True.
    compute : str or list of str, optional
        Which functionals to compute. ``'standard'`` returns the 14 base
        functionals; ``'all'`` additionally computes ``w103``, ``w104``,
        spherical Minkowski summaries (``msm_ql``, ``msm_wl``), and derived
        scalars (``*_beta``, ``*_trace``, ``*_trace_ratio``); a list of names
        selects specific quantities.
    compute_eigensystems : bool, optional
        If False, eigenvalues and eigenvectors for rank-2 tensors are skipped
        (``*_eigvals`` / ``*_eigvecs`` keys are omitted). Default is True.
    return_count : bool, optional
        If True, include the total number of connected mesh components
        ``n_objects`` in the return tuple. Default is False.
    autolabel : bool, optional
        If False (default), treat the entire extracted isosurface as a single
        body and return a flat dict matching ``minkowski_tensors(..., labels=None)``.
        If True, detect connected mesh components automatically (equivalent to
        ``minkowski_tensors(..., labels='auto')``) and return a nested dict
        keyed by 1-based component index (``1``, ``2``, ...). Note that
        grouping is by connected *surface* component rather than 3D solid
        region, so an internal cavity of a hollow object is separated as its
        own component with inward-facing normals (negative ``w000``).
    pad : bool, optional
        If True (default), apply a 1-voxel exterior border on all six faces of
        ``field`` before calling ``marching_cubes`` so objects touching the
        array boundary produce closed (non-open) surfaces. Vertex coordinates
        are shifted back to the original image coordinate system. Set
        ``pad=False`` if the volume is already padded or open surfaces at the
        boundary are intentional.
    pad_value : float or None, optional
        Constant value used for the 1-voxel border when ``pad=True``. Ignored
        when ``pad=False``. When ``None`` (default), automatically selects
        ``field.min()`` if it is strictly below ``level`` (else ``level - 1.0``)
        for ``interior='above'``, or ``field.max()`` if it is strictly above
        ``level`` (else ``level + 1.0``) for ``interior='below'``. For
        boundary-touching objects, linear interpolation across the padded
        border depends on ``pad_value``; for unbounded SDFs or intensity
        volumes whose global extrema vary with volume extent, pass an explicit
        ``pad_value`` for consistent boundary-cap placement.
    msm_max_l : int, optional
        Maximum spherical harmonic order for spherical Minkowski tensors.
        Default is 8. See ``minkowski_tensors`` for details.
    return_meshes : bool, optional
        If True, also return the mesh(es) produced internally by
        ``marching_cubes`` in original image physical coordinates (no extra
        ``marching_cubes`` call is made).

        When ``autolabel=False``: ``meshes`` is a ``(verts, faces)`` tuple.

        When ``autolabel=True``: ``meshes`` is a ``dict[int, (verts, faces)]``
        keyed identically to ``results`` (compatible with
        ``combine_label_meshes``). Default is False.

    Returns
    -------
    dict or dict[int, dict]
        When ``return_count=False`` and ``return_meshes=False``: a flat dict of
        Minkowski functionals when ``autolabel=False``, or a nested dict keyed
        by 1-based component index when ``autolabel=True``. When
        ``autolabel=True`` and no isosurface exists at ``level``, returns ``{}``.
    tuple
        When ``return_count`` and/or ``return_meshes`` is True, the results
        above are augmented in this order:

        ``(results, n_objects)`` when only ``return_count=True``.

        ``(results, meshes)`` when only ``return_meshes=True``.

        ``(results, n_objects, meshes)`` when both are True.

    Raises
    ------
    ValueError
        If ``field`` is complex, not a non-empty 3D finite array, or has an
        axis shorter than 2 when ``pad=False``; if ``level`` or ``pad_value``
        is non-finite, ``interior`` is invalid, ``spacing`` is not a 3-tuple of
        positive finite floats, ``center='centroid_voxel'``, or (when
        ``autolabel=False``) no isosurface exists at ``level``.

    Notes
    -----
    Requires *scikit-image*.

    **Calibration caveat for probability images:** Sub-voxel accuracy gains
    over thresholding + ``minkowski_tensors_from_label_image`` rely on the
    probability field transitioning smoothly across 2--3 voxels near the
    boundary. Over-confident segmentation networks whose probabilities saturate
    sharply at 0 and 1 behave like binary masks and will yield little or no
    geometric improvement.

    Examples
    --------
    Probability map (interior > 0.5):

    >>> res = minkowski_tensors_from_scalar_field(prob_map, level=0.5)

    Signed distance function (interior < 0.0):

    >>> res = minkowski_tensors_from_scalar_field(sdf, level=0.0, interior='below')
    """
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        raise ImportError(
            "scikit-image is required for minkowski_tensors_from_scalar_field. "
            "Install it with: pip install scikit-image"
        )

    if interior not in ('above', 'below'):
        raise ValueError(
            f"interior must be 'above' or 'below', got {interior!r}"
        )
    if isinstance(center, str) and center == 'centroid_voxel':
        raise ValueError(
            "'centroid_voxel' is not supported for scalar fields; "
            "use 'centroid_mesh', 'reference_centroid', a (3,) coordinate array, or None."
        )

    field_arr = np.asarray(field)
    if np.iscomplexobj(field_arr):
        raise ValueError(
            f"field must be real-valued, got complex dtype {field_arr.dtype}"
        )
    if not (np.issubdtype(field_arr.dtype, np.number) or field_arr.dtype == np.bool_):
        field_arr = np.asarray(field_arr, dtype=np.float64)
    if field_arr.ndim != 3:
        raise ValueError(f"field must be a 3D array, got shape {field_arr.shape}")
    if field_arr.size == 0:
        raise ValueError("field must not be empty")
    if not pad and any(s < 2 for s in field_arr.shape):
        raise ValueError(
            f"field must have shape >= (2, 2, 2) when pad=False, got shape {field_arr.shape}"
        )
    if np.issubdtype(field_arr.dtype, np.floating) and not np.all(np.isfinite(field_arr)):
        raise ValueError("field must contain only finite values (no NaN or Inf)")

    level = float(level)
    if not np.isfinite(level):
        raise ValueError(f"level must be finite, got {level}")

    spacing = tuple(float(s) for s in spacing)
    if len(spacing) != 3 or not all(np.isfinite(s) and s > 0 for s in spacing):
        raise ValueError(
            f"spacing must be a 3-tuple of positive finite floats, got {spacing}"
        )

    raw_min = float(field_arr.min())
    raw_max = float(field_arr.max())

    def _empty_return():
        if not autolabel:
            raise ValueError(
                f"No isosurface found at level={level} for field with range "
                f"[{raw_min}, {raw_max}] (interior={interior!r})."
            )
        if return_count and return_meshes:
            return {}, 0, {}
        if return_count:
            return {}, 0
        if return_meshes:
            return {}, {}
        return {}

    # Determine optional 1-voxel exterior padding value
    pad_val = None
    pad_strictly_interior = False
    if pad:
        if pad_value is not None:
            pad_val = float(pad_value)
            if not np.isfinite(pad_val):
                raise ValueError(f"pad_value must be finite, got {pad_value}")
            if (interior == 'above' and pad_val >= level) or (
                interior == 'below' and pad_val <= level
            ):
                warnings.warn(
                    f"pad_value={pad_val} is not strictly on the exterior side "
                    f"of level={level} for interior={interior!r}; boundary-touching "
                    "objects may not close as expected.",
                    UserWarning,
                    stacklevel=2,
                )
                pad_strictly_interior = (
                    (interior == 'above' and pad_val > level)
                    or (interior == 'below' and pad_val < level)
                )
        elif interior == 'above':
            pad_val = raw_min if raw_min < level else level - 1.0
        else:
            pad_val = raw_max if raw_max > level else level + 1.0

    # Derive padded bounds in O(1) and check crossing in canonical (interior > work_level) coords
    padded_min = min(raw_min, pad_val) if pad else raw_min
    padded_max = max(raw_max, pad_val) if pad else raw_max
    if interior == 'below':
        work_min = -padded_max
        work_max = -padded_min
        work_level = -level
    else:
        work_min = padded_min
        work_max = padded_max
        work_level = level

    if not (work_min <= work_level < work_max):
        return _empty_return()

    sp_arr = np.array(spacing, dtype=np.float64)

    # Crop to the 1-voxel-dilated bounding box of interior voxels BEFORE
    # float64 conversion, padding, and negation to avoid full-volume copies.
    if not pad_strictly_interior:
        interior_mask = (field_arr > level) if interior == 'above' else (field_arr < level)
        z_idx = np.flatnonzero(interior_mask.any(axis=(1, 2)))
        y_idx = np.flatnonzero(interior_mask.any(axis=(0, 2)))
        x_idx = np.flatnonzero(interior_mask.any(axis=(0, 1)))
        if len(z_idx) == 0 or len(y_idx) == 0 or len(x_idx) == 0:
            return _empty_return()

        zb0, zb1 = int(z_idx[0]), int(z_idx[-1])
        yb0, yb1 = int(y_idx[0]), int(y_idx[-1])
        xb0, xb1 = int(x_idx[0]), int(x_idx[-1])

        fz0 = max(0, zb0 - 1)
        fy0 = max(0, yb0 - 1)
        fx0 = max(0, xb0 - 1)
        fz1 = min(field_arr.shape[0], zb1 + 2)
        fy1 = min(field_arr.shape[1], yb1 + 2)
        fx1 = min(field_arr.shape[2], xb1 + 2)

        crop = np.asarray(field_arr[fz0:fz1, fy0:fy1, fx0:fx1], dtype=np.float64)
        if pad:
            pad_z = (1 if zb0 == 0 else 0, 1 if zb1 == field_arr.shape[0] - 1 else 0)
            pad_y = (1 if yb0 == 0 else 0, 1 if yb1 == field_arr.shape[1] - 1 else 0)
            pad_x = (1 if xb0 == 0 else 0, 1 if xb1 == field_arr.shape[2] - 1 else 0)
            if any(pad_z) or any(pad_y) or any(pad_x):
                crop = np.pad(
                    crop,
                    pad_width=(pad_z, pad_y, pad_x),
                    mode='constant',
                    constant_values=pad_val,
                )
            crop_origin = np.array(
                [fz0 - pad_z[0], fy0 - pad_y[0], fx0 - pad_x[0]], dtype=np.float64
            ) * sp_arr
        else:
            crop_origin = np.array([fz0, fy0, fx0], dtype=np.float64) * sp_arr
    else:
        # Rare warned path: explicit pad_value lies strictly on the interior side,
        # so the 1-voxel border itself is interior and cannot be cropped out.
        crop = np.pad(
            np.asarray(field_arr, dtype=np.float64),
            pad_width=1,
            mode='constant',
            constant_values=pad_val,
        )
        crop_origin = -sp_arr

    # Canonicalize so the solid interior is always strictly > work_level.
    # This makes interior='above' and interior='below' exact mathematical duals
    # and ensures marching_cubes(..., gradient_direction='ascent') always emits
    # outward-facing right-hand-rule normals (even for open surfaces and shells).
    if interior == 'below':
        crop = -crop

    try:
        verts, faces, _, _ = marching_cubes(
            crop,
            level=work_level,
            spacing=spacing,
            gradient_direction='ascent',
            allow_degenerate=False,
        )
    except (ValueError, RuntimeError):
        return _empty_return()

    if len(faces) == 0:
        return _empty_return()

    verts = np.asarray(verts, dtype=np.float64) + crop_origin
    faces = np.asarray(faces, dtype=np.int64)

    # marching_cubes(..., allow_degenerate=False) may drop degenerate faces
    # without pruning newly unreferenced vertices; compact to prevent orphan
    # vertices from emitting spurious curvature warnings or gapped meshes.
    used_verts = np.unique(faces)
    if len(used_verts) < len(verts):
        remap = np.full(len(verts), -1, dtype=np.int64)
        remap[used_verts] = np.arange(len(used_verts), dtype=np.int64)
        verts = verts[used_verts]
        faces = remap[faces]

    face_labels_arr = _label_mesh_components(faces) if autolabel else None
    mt_result = minkowski_tensors(
        verts,
        faces,
        labels=face_labels_arr,
        center=center,
        center_per_label=center_per_label,
        compute=compute,
        compute_eigensystems=compute_eigensystems,
        return_count=return_count,
        msm_max_l=msm_max_l,
    )

    if not return_meshes:
        return mt_result

    if return_count:
        results, n_objects = mt_result
    else:
        results = mt_result

    if autolabel:
        meshes = {}
        for comp in np.unique(face_labels_arr):
            comp_faces_global = faces[face_labels_arr == comp]
            used_idx = np.unique(comp_faces_global)
            remap = np.full(len(verts), -1, dtype=np.int64)
            remap[used_idx] = np.arange(len(used_idx), dtype=np.int64)
            meshes[int(comp)] = (verts[used_idx], remap[comp_faces_global])
    else:
        meshes = (verts, faces)

    if return_count:
        return results, n_objects, meshes
    return results, meshes


def combine_label_meshes(label_meshes, color_by=None):
    """Concatenate per-label meshes into a single mesh for visualization.

    Parameters
    ----------
    label_meshes : dict[int, (verts, faces)]
        E.g. from ``minkowski_tensors_from_label_image(..., return_meshes=True)``.
    color_by : None or dict[int, float], optional
        ``None`` (default): vertex values are the label id (discrete
        per-object coloring).
        ``dict``: label -> scalar (e.g. ``{lbl: r['w220_beta'] for lbl, r in
        results.items()}``), broadcast across that label's vertices.

    Returns
    -------
    verts : (N, 3) float64 array
    faces : (M, 3) int64 array
    values : (N,) float64 array
        Ready to unpack into ``viewer.add_surface((verts, faces, values))``.
    vertex_labels : (N,) int64 array
        Label id owning each vertex (for picking/debugging/QA; not part of
        the ``add_surface`` tuple).
    """
    labels_sorted = sorted(label_meshes.keys())
    all_verts, all_faces, all_values, all_vertex_labels = [], [], [], []
    vert_offset = 0
    for lab in labels_sorted:
        v, f = label_meshes[lab]
        v = np.asarray(v, dtype=np.float64)
        f = np.asarray(f, dtype=np.int64)
        all_verts.append(v)
        all_faces.append(f + vert_offset)
        if color_by is None:
            val = float(lab)
        else:
            try:
                val = float(color_by[lab])
            except KeyError:
                raise KeyError(
                    f"color_by is missing a value for label {lab}; "
                    f"it must have an entry for every key in label_meshes."
                ) from None
        all_values.append(np.full(len(v), val, dtype=np.float64))
        all_vertex_labels.append(np.full(len(v), lab, dtype=np.int64))
        vert_offset += len(v)
    if all_verts:
        verts = np.vstack(all_verts)
        faces = np.vstack(all_faces)
        values = np.concatenate(all_values)
        vertex_labels = np.concatenate(all_vertex_labels)
    else:
        verts = np.empty((0, 3), dtype=np.float64)
        faces = np.empty((0, 3), dtype=np.int64)
        values = np.empty((0,), dtype=np.float64)
        vertex_labels = np.empty((0,), dtype=np.int64)
    return verts, faces, values, vertex_labels
