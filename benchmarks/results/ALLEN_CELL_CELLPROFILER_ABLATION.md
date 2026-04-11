# Allen Cell Nuclei: CellProfiler Feature Ablation

## Overview

Ablation of CellProfiler features into shape-only, position-only, and full sets.
CellProfiler provides 22 features per nucleus: 8 pure shape descriptors and
14 position/bounding-box/count columns.

**Classifier**: LinearSVC (liblinear)
**Dataset**: Non-rotated nuclei only (`nuclei/`)
**Optimization**: BayesSearchCV, n_iter=20, 5-fold stratified CV
**Evaluation**: 3 seeds, balanced accuracy + geometric mean
**Input**: `cellprofiler_features.csv` (22 features, joined by `image_num`)

---

## Feature Breakdown

### Shape features (8)

| Feature | Description |
|---------|-------------|
| AreaShape_EquivalentDiameter | Diameter of a sphere with the same volume |
| AreaShape_EulerNumber | Topological genus (holes) |
| AreaShape_Extent | Volume / bounding box volume |
| AreaShape_MajorAxisLength | Length of major principal axis |
| AreaShape_MinorAxisLength | Length of minor principal axis |
| AreaShape_Solidity | Volume / convex hull volume |
| AreaShape_SurfaceArea | Surface area |
| AreaShape_Volume | Voxel volume |

### Position / bounding-box / count features (14)

| Feature | Description |
|---------|-------------|
| Image_Count_ConvertImageToObjects | Object count per image |
| AreaShape_BoundingBoxMaximum_X/Y/Z | Bounding box upper corners (3) |
| AreaShape_BoundingBoxMinimum_X/Y/Z | Bounding box lower corners (3) |
| AreaShape_BoundingBoxVolume | Bounding box volume |
| AreaShape_Center_X/Y/Z | Object centroid (3) |
| Location_Center_X/Y/Z | Image-level centroid (3) |

---

## Results

| Feature Set | # Feat | Balanced Accuracy | Geo. Mean | Best C | Best PCA |
|-------------|--------|-------------------|-----------|--------|----------|
| CellProfiler (position only) | 14 | 0.609 ± 0.001 | 0.535 ± 0.003 | 225 | 14 |
| CellProfiler (shape only) | 8 | 0.738 ± 0.008 | 0.727 ± 0.010 | 1000 | 8 |
| CellProfiler (full) | 22 | **0.769 ± 0.003** | **0.761 ± 0.003** | 225 | 22 |

---

## Interpretation

### Shape features carry the majority of the signal

With only 8 features, shape-only scores 0.738 — capturing 96% of the full set's performance
(0.769) using 36% of the features. The 8 descriptors (volume, surface area, axis lengths,
solidity, extent, Euler number) are highly informative about mitotic stage because nuclear
morphology changes dramatically across the cell cycle: nuclei expand, elongate, and divide
in characteristic ways at each stage.

### Position features add real but smaller signal

Position-only (14 features, 0.609) is well above chance (0.167 on 6 classes), reflecting
genuine spatial organisation of mitotic stages within the imaging volume — nuclei at
different cell-cycle stages occupy statistically distinct spatial zones in the dataset.
However, at 0.609 it is substantially below shape-only (0.738): spatial location is a
weaker discriminator than morphological shape for this task.

### Full set gains come from combining orthogonal signals

The full 22-feature set (0.769) exceeds both subsets — it is not simply dominated by shape:

| Comparison | Δ Bal. Acc |
|---|---|
| Position only → Full | +16.0 pp |
| Shape only → Full | +3.1 pp |
| Shape only → Position only | −12.9 pp |

The +3.1 pp gain from adding position to shape (vs the +16.0 pp gain from adding shape to
position) confirms that shape is the primary signal source. Position encodes complementary
spatial information not captured by morphology alone.

Both the full set and position-only use C=225 with full PCA retention, indicating that
all 14 position features contribute and there is no redundancy to regularise away.
Shape-only uses C=1000 — slightly higher, likely because removing the two centroid
representations (AreaShape_Center and Location_Center, which partially overlap) eliminates
mild redundancy from the full feature set.

### CellProfiler in the broader benchmark context

| Feature Set | # Feat | Bal. Acc | Notes |
|---|---|---|---|
| CellProfiler (position only) | 14 | 0.609 | Below all Minkowski/invariant sets |
| CellProfiler (shape only) | 8 | 0.738 | Beats Minkowski (tensors) (62 feat, 0.746)? No — slightly below |
| CellProfiler (full) | 22 | 0.769 | Beats Minkowski (tensors) (0.746), below Eigenvalues only (0.791) |

CellProfiler (shape only) at 0.738 sits just below Minkowski (tensors) at 0.746 — a gap of
only 0.8 pp despite using 8 vs 62 features. This demonstrates that the 8 hand-crafted
geometric descriptors are nearly as informative as the full raw Minkowski tensor representation,
and far more compact.
CellProfiler (full) at 0.769 beats Minkowski (tensors) by +2.3 pp, but is overtaken by any
eigenvalue-augmented Minkowski set (Eigenvalues only: 0.791, +2.2 pp over CellProfiler full).

---

## Configuration

```bash
# All three conditions
python benchmarks/invariants_classification.py \
    --input ../Minkowski_classifier/data/allen_cell/mitotic_cells_annotated/nuclei/minkowski_tensors_with_eigen_vals.csv \
    --cellprofiler-input ../Minkowski_classifier/data/allen_cell/mitotic_cells_annotated/nuclei/cellprofiler_features.csv \
    --output benchmarks/results/allen_cell_nuclei_cellprofiler_ablation \
    --include "CellProfiler" \
    --optimize \
    --n_iter 20 \
    --linear-only \
    --seeds 3 \
    --n_jobs 5
```

---

## Runtime

- All three conditions combined: ~1.5 min (CellProfiler features are pre-computed; cost is BayesSearchCV only)
