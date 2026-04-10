# Changelog

All notable changes to this project will be documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Version numbers follow [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added
- `pykarambola.invariants` module for computing SO(3), O(3), and SO(2) rotational invariants from Minkowski tensors (#102)
  - `compute_invariants(tensors_dict, max_degree, symmetry, deduplicate_scalars)` computes a complete basis of polynomial invariants up to degree 3
  - `decompose_all(tensors_dict)` decomposes tensors into irreducible representations (trace/traceless split for rank-2)
  - Flexible input: accepts any combination of rank 0, 1, 2 tensors (not limited to the 14 standard Minkowski tensors)
  - Supports O(3) (true scalars), SO(3) (includes pseudo-scalars), and SO(2) (z-rotation only) symmetry groups
  - `symmetry='SO2'` decomposes tensors by SO(2) charge m and constructs invariants from m=0 scalars, |m|=1 doublets, and |m|=2 doublets up to degree 3; yields 754 invariants for the 14 standard Minkowski tensors
  - Automatic deduplication of linearly dependent scalars (Tr(w102)/3 = w100, Tr(w202)/3 = w200)

---

## [0.2.0] - Unreleased

### Added
- `center='centroid_mesh'` option in `minkowski_tensors` (volume-weighted center of mass = w010/w000), consistent with the existing option in `minkowski_tensors_from_label_image` (#73)
- `center_per_label=True` boolean parameter in both `minkowski_tensors` and `minkowski_tensors_from_label_image` to control whether the centroid is computed independently per label (`True`, default) or from the full mesh (`False`) (#73)
- `return_count=False` flag in both `minkowski_tensors` and `minkowski_tensors_from_label_image`; when `True`, returns `(results, n_objects)` where `n_objects` is the total number of connected components (#80)
- `labels='auto'` option in `minkowski_tensors()` to automatically detect connected mesh components and return results keyed by 1-based component index
- `autolabel=False` parameter in `minkowski_tensors_from_label_image()`: when `True`, treats the label image as binary, builds one mesh from all non-zero voxels, and detects connected components automatically (#80)
- `_label_mesh_components()` public helper function that returns per-face component labels for a triangular mesh using scipy-based connected components detection

### Changed
- `center='centroid'` in `minkowski_tensors` renamed to `center='reference_centroid'` to match the C++ `--reference_centroid` flag (#73)

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

[Unreleased]: https://github.com/Pitt-IshiharaLab/pykarambola/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Pitt-IshiharaLab/pykarambola/releases/tag/v0.1.0
