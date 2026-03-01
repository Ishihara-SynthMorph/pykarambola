# pykarambola

Python implementation of [karambola](https://github.com/morphometry/karambola) — a package for computing **Minkowski functionals and tensors** on 3D triangulated surfaces.

Minkowski functionals (volume, surface area, integrated mean curvature, Euler characteristic) and their tensor generalizations characterize the geometry and morphology of 3D shapes. They are widely used in structural analysis, materials science, and computational physics.

## New in pykarambola

Compared to the original C++ karambola, this Python port adds:

- **OBJ and GLB parsers** — read Wavefront OBJ and binary glTF (`.glb`) meshes directly, in addition to the original `.poly` and `.off` formats.
- **High-level API** — `minkowski_functionals()` accepts NumPy arrays and returns a plain dict, making it easy to integrate into pipelines without dealing with the lower-level triangulation types.
- **Label-image API** — `minkowski_functionals_from_label_image()` extracts surfaces from a 3D integer label image via marching cubes and computes functionals for every label in one call.

## Requirements

- Python ≥ 3.9
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)

**Optional:**
- [Cython](https://cython.org/) ≥ 3.0 — compiled C acceleration (`pip install "pykarambola[accel]"`)
- [scikit-image](https://scikit-image.org/) — label-image API (`pip install "pykarambola[dev]"`)
- [trimesh](https://trimesh.org/) — GLB/glTF file support (`pip install "pykarambola[glb]"`)

## Installation

```bash
pip install pykarambola
```

For optional Cython acceleration:

```bash
pip install "pykarambola[accel]"
```

For development (includes pytest and scikit-image):

```bash
pip install "pykarambola[dev]"
```

GLB/glTF support requires [trimesh](https://trimesh.org/):

```bash
pip install "pykarambola[glb]"
```

## High-level API

### From NumPy arrays

`minkowski_functionals()` is the main entry point. Pass vertices and faces as NumPy arrays and get back a plain dict:

```python
import pykarambola as pk

result = pk.minkowski_functionals(
    verts,   # (V, 3) float64 array of vertex positions
    faces,   # (F, 3) int64 array of vertex indices
)

print(result["w000"])   # volume
print(result["w100"])   # surface area
print(result["w200"])   # integrated mean curvature
print(result["w300"])   # Euler characteristic
print(result["w020"])   # 3×3 Minkowski tensor
print(result["w020_eigvals"])   # eigenvalues of w020
print(result["w020_eigvecs"])   # eigenvectors of w020 (columns)
```

Control which quantities are computed with the `compute` argument:

```python
# default: 14 standard functionals + eigensystems for rank-2 tensors
result = pk.minkowski_functionals(verts, faces, compute="standard")

# include higher-order tensors (w103, w104) and spherical Minkowski metrics
result = pk.minkowski_functionals(verts, faces, compute="all")

# compute only specific quantities
result = pk.minkowski_functionals(verts, faces, compute=["w000", "w100", "w020"])
```

### From a 3D label image

`minkowski_functionals_from_label_image()` takes a 3D integer array, runs marching cubes on each label, and returns a dict of results keyed by label value. Requires [scikit-image](https://scikit-image.org/).

```python
import numpy as np
import pykarambola as pk

label_image = np.zeros((64, 64, 64), dtype=int)
label_image[10:40, 10:40, 10:40] = 1
label_image[40:60, 40:60, 40:60] = 2

result = pk.minkowski_functionals_from_label_image(
    label_image,
    spacing=(0.5, 0.5, 0.5),   # voxel size in physical units
    center="centroid",          # shift tensors to per-label centroid
)

print(result[1]["w000"])   # volume of label 1
print(result[2]["w100"])   # surface area of label 2
```

### Multi-label meshes

Pass per-face integer labels to compute functionals for multiple bodies in a single mesh:

```python
result = pk.minkowski_functionals(verts, faces, labels=face_labels)
# result is dict[int, dict]
print(result[1]["w000"])
print(result[2]["w000"])
```

## File I/O

pykarambola can read four mesh formats. The parsers return a `Triangulation` object whose `.vertices` and `.triangles` arrays can be passed directly to `minkowski_functionals()`.

```python
surface = pk.parse_poly_file("my_surface.poly")   # karambola native
surface = pk.parse_off_file("my_surface.off")     # Object File Format
surface = pk.parse_obj_file("my_surface.obj")     # Wavefront OBJ  (new)
surface = pk.parse_glb_file("my_surface.glb")     # binary glTF    (new, requires trimesh)

result = pk.minkowski_functionals(surface.vertices, surface.triangles)
```

| Extension | Description |
|-----------|-------------|
| `.poly`   | karambola native format |
| `.off`    | Object File Format |
| `.obj`    | Wavefront OBJ |
| `.glb`    | GL Transmission Format (binary glTF) — requires `trimesh` |

## Command-line interface

```
python -m pykarambola [options] <surface_file>
```

Supported input formats: `.poly`, `.off`, `.obj`, `.glb`.

## Computed quantities

| Name | Type | Description |
|------|------|-------------|
| `w000` | scalar | Volume |
| `w100` | scalar | Surface area |
| `w200` | scalar | Integrated mean curvature |
| `w300` | scalar | Euler characteristic |
| `w010` | vector | Minkowski vector (volume) |
| `w110` | vector | Minkowski vector (surface) |
| `w210` | vector | Minkowski vector (curvature) |
| `w310` | vector | Minkowski vector (topology) |
| `w020` | rank-2 tensor | Minkowski tensor (volume) |
| `w120` | rank-2 tensor | Minkowski tensor (surface) |
| `w220` | rank-2 tensor | Minkowski tensor (curvature) |
| `w320` | rank-2 tensor | Minkowski tensor (topology) |
| `w102` | rank-2 tensor | Minkowski tensor (surface, normal-normal) |
| `w202` | rank-2 tensor | Minkowski tensor (curvature, normal-normal) |
| `w103` | rank-3 tensor | Higher-order tensor |
| `w104` | rank-4 tensor | Higher-order tensor |
| `msm_ql`, `msm_wl` | arrays | Minkowski structure metrics (spherical) |

Rank-2 tensors additionally yield `{name}_eigvals` and `{name}_eigvecs` entries.

## Citation

If you use pykarambola in published work, please cite the original karambola package:

> Schaller, F. M., Kapfer, S. C., & Schröder-Turk, G. E.
> *karambola — 3D Minkowski Tensor Package* (v2.0).
> https://github.com/morphometry/karambola

See [`CITATION.cff`](CITATION.cff) for machine-readable citation metadata.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for development setup, Git workflow, versioning, and release instructions.

## License

See [`LICENSE`](LICENSE).
