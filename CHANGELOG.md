# Changelog

All notable changes to this project will be documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Version numbers follow [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

*(Add entries here as you work. Move them to a versioned section when releasing.)*

---

## [0.1.0] - 2026-02-25

Initial Python port of [karambola](https://github.com/morphometry/karambola).

### Added

- Core Minkowski functional and tensor compute functions (`calculate_w*`) ported from C++ karambola
- `Triangulation` data structure with precomputed per-face normals, edge lengths, and dihedral angles
- High-level NumPy array API: `minkowski_functionals(verts, faces)` returns a plain dict
- Label-image API: `minkowski_functionals_from_label_image()` runs marching cubes per label and returns results keyed by label value
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
