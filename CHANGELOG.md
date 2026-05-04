# Changelog

All notable changes to this project will be documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Version numbers follow [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

---

## [0.3.0] - 2026-04-30

### Added
- `minkowski_tensors` now accepts a `Triangulation` object as its first argument; `faces` may be omitted. Works with objects returned by all four parsers (`parse_obj_file`, `parse_off_file`, `parse_poly_file`, `parse_glb_file`). Labels embedded in the triangulation are extracted automatically when they carry meaningful per-body information. (#90)
- `pad=True` parameter in `minkowski_tensors_from_label_image`: applies a 1-voxel zero-padding on all six faces before calling `marching_cubes`, ensuring that objects touching the array boundary produce closed (non-open) surfaces. Vertex coordinates and `centroid_voxel` computations are corrected for the offset. Set `pad=False` to restore prior behaviour. (#99)

### Changed
- `minkowski_tensors` now emits a `UserWarning` for **open surfaces** (meshes with boundary edges) and sets `w000` and `w020` to `NaN` for the affected labels, matching C++ karambola's behaviour. (#94)
- `minkowski_tensors` now emits a `UserWarning` for **non-manifold meshes** (more permissive than the C++ CLI, which aborts). (#94)
- `minkowski_tensors_from_label_image` now pads the label image with a 1-voxel zero border by default (`pad=True`). Callers whose inputs already include padding, or who intentionally want open surfaces at the boundary, should pass `pad=False` to restore prior behaviour. (#99)
- Dependency lower bounds added: `numpy>=1.22`, `scipy>=1.8`; `scikit-image>=0.19` in the `dev` and `notebooks` extras; `numpy<2.0` upper bound added to the `[build-system]` requires. (#64)

### Fixed
- `np.asarray` calls for `labels` and face-index arrays now specify `dtype=np.int64` explicitly, preventing silent coercion when callers pass float-dtype label arrays. (#64)

### Documentation
- Added `Examples` sections to `parse_obj_file`, `parse_off_file`, `parse_poly_file`, and `parse_glb_file` docstrings (numpydoc conformance). (#64)
- Added `Notes` section to `calculate_eigensystem` clarifying its role relative to the high-level API. (#64)
- Improved error message in `parse_poly_file` to report the actual token count when vertex coordinates are missing. (#64)
- README and manuscript: documented `labels='auto'`, `return_count`, `_beta`/`_trace`/`_trace_ratio` derived quantities, `compute_eigensystems`, and `center_per_label` — all previously undocumented. Added `labels='auto'` code example, `center` argument reference table, quantities-table preset annotation, and CLI `--help` note. (#95)

---

## [0.2.0] - 2026-03-19

### Added
- `compute_eigensystems=False` parameter in `minkowski_tensors` and `minkowski_tensors_from_label_image`: skips eigendecomposition for rank-2 tensors (`*_eigvals` / `*_eigvecs` keys are omitted), reducing runtime for batch jobs that do not need eigensystems (#51)
- `{name}_beta` anisotropy index (ratio of smallest to largest eigenvalue magnitude) for each rank-2 tensor (#1)
- `{name}_trace` (matrix trace) and `{name}_trace_ratio` (trace divided by the corresponding Minkowski scalar, e.g. `Tr(w020)/w000`) for each rank-2 tensor; `_trace_ratio` is defined only for the wX20 family (#2)
- `center='centroid_mesh'` option in `minkowski_tensors` (volume-weighted center of mass via divergence theorem), consistent with the existing option in `minkowski_tensors_from_label_image` (#73)
- `center_per_label=True` boolean parameter in both `minkowski_tensors` and `minkowski_tensors_from_label_image` to control whether the centroid is computed independently per label (`True`, default) or from the full mesh (`False`) (#73)
- `return_count=False` flag in both `minkowski_tensors` and `minkowski_tensors_from_label_image`; when `True`, returns `(results, n_objects)` where `n_objects` is the total number of connected components (#80)
- `labels='auto'` option in `minkowski_tensors()` to automatically detect connected mesh components and return results keyed by 1-based component index (#80)
- `autolabel=False` parameter in `minkowski_tensors_from_label_image()`: when `True`, treats the label image as binary, builds one mesh from all non-zero voxels, and detects connected components automatically (#80)

### Changed
- All API symbols and documentation renamed from "Minkowski functionals" to "Minkowski tensors" throughout the codebase (#3)
- Eigenvalues and eigenvectors are now sorted by ascending eigenvalue magnitude, matching C++ karambola behaviour (#59)
- `minkowski_tensors_from_label_image` now uses `gradient_direction='ascent'` in marching cubes and computes centroid references from mesh geometry rather than voxel coordinates (#4)
- `center='centroid'` in `minkowski_tensors` renamed to `center='reference_centroid'` to match the C++ `--reference_centroid` flag (#73)
- Prerequisite scalars and vectors are now skipped in non-centroid mode; only quantities that are actually needed are computed (#47)

### Fixed
- Zero-denominator guards added to `get_ref_vec`, `calculate_w300`, and `calculate_w320` to prevent silent `ZeroDivisionError` on flat, open, or toroidal surfaces (#42, #43, #49)
- Degenerate (zero-area) triangles now emit a `UserWarning` instead of producing spurious normals that corrupt dihedral-angle calculations (#44)
- Unknown names in `compute=[…]` now raise `ValueError` instead of being silently ignored (#45)
- `center` argument now validated for shape `(3,)` before use; invalid shapes raise `ValueError` (#46)
- `minkowski_tensors` return type is now consistently `dict[int, dict]` whenever `labels` is provided, even for a single unique label (#48)
- `minkowski_tensors_from_label_image` now enforces integer dtype on the input array, emitting a `UserWarning` when conversion is required (#52)
- `_EXTRA` derived quantities (`_beta`, `_trace`, `_trace_ratio`) dependency-chain bug fixed: parent tensors are now correctly promoted into `wanted` before computation (#53)

---

## [0.1.0] - 2026-02-25

Initial Python port of [karambola](https://github.com/morphometry/karambola).

### Added

- Core Minkowski tensor compute functions (`calculate_w*`) ported from C++ karambola
- `Triangulation` data structure with precomputed per-face normals, edge lengths, and dihedral angles
- High-level NumPy array API: `minkowski_tensors(verts, faces)` returns a plain dict
- Label-image API: `minkowski_tensors_from_label_image()` runs marching cubes per label and returns results keyed by label value
- Rank-2 tensor eigensystem decomposition (`_eigvals`, `_eigvecs` entries in results)
- File format parsers: `.poly` (karambola native), `.off`, `.obj` (new), `.glb` / binary glTF (new, requires `trimesh`)
- Command-line interface: `python -m pykarambola`
- Optional Cython acceleration (`pip install "pykarambola[accel]"`)
- CI via GitHub Actions: pytest on Ubuntu, macOS, and Windows across Python 3.9, 3.11, and 3.13
- Slack notifications for CI failures on `main` and PR merges
- `pykarambola.__version__` attribute (reads from package metadata at runtime)
- `CONTRIBUTING.md` developer guide covering dev setup, Git workflow, versioning, and release steps
- GitHub Actions workflow to automatically add new issues to the project board

---

[Unreleased]: https://github.com/Ishihara-SynthMorph/pykarambola/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/Ishihara-SynthMorph/pykarambola/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Ishihara-SynthMorph/pykarambola/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Ishihara-SynthMorph/pykarambola/releases/tag/v0.1.0
