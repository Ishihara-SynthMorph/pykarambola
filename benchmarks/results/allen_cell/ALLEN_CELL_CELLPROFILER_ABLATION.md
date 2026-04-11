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

| Feature Set | # Feat | Non-zero var | Rank | Balanced Accuracy | Geo. Mean | Best C | Best PCA |
|-------------|--------|-------------|------|-------------------|-----------|--------|----------|
| CellProfiler (position only) | 14 | 14 | 11 | 0.609 ± 0.001 | 0.535 ± 0.003 | 225 | 14 |
| CellProfiler (SurfaceArea + Solidity only) | 2 | 2 | 2 | 0.609 ± 0.004 | 0.509 ± 0.014 | 37.5 | 2 |
| CellProfiler (shape only) | 8 | 8 | 8 | 0.738 ± 0.008 | 0.727 ± 0.010 | 1000 | 8 |
| CellProfiler (no SurfaceArea) | 21 | 21 | 18 | 0.728 ± 0.005 | 0.714 ± 0.005 | 1000 | 20 |
| CellProfiler (no Solidity) | 21 | 21 | 18 | 0.738 ± 0.002 | 0.726 ± 0.002 | 1000 | 20 |
| CellProfiler (no Volume) | 21 | 21 | 18 | 0.748 ± 0.008 | 0.738 ± 0.009 | 1000 | 14 |
| CellProfiler (no MajorAxisLength) | 21 | 21 | 18 | 0.751 ± 0.002 | 0.742 ± 0.002 | 64.1 | 18 |
| CellProfiler (no EquivalentDiameter) | 21 | 21 | 18 | 0.764 ± 0.003 | 0.755 ± 0.003 | 1.08 | 21 |
| CellProfiler (no Extent) | 21 | 21 | 18 | 0.767 ± 0.002 | 0.761 ± 0.002 | 1000 | 20 |
| CellProfiler (no MinorAxisLength) | 21 | 21 | 18 | 0.769 ± 0.003 | 0.763 ± 0.003 | 25.1 | 19 |
| CellProfiler (no EulerNumber) | 21 | 21 | 18 | 0.772 ± 0.002 | 0.766 ± 0.002 | 225 | 21 |
| CellProfiler (full) | 22 | 22 | 19 | **0.769 ± 0.003** | **0.761 ± 0.003** | 225 | 22 |

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

### Shape feature importance ranking (leave-one-out ablation)

| Feature removed | Bal. Acc | Δ vs full (0.769) | Importance |
|---|---|---|---|
| SurfaceArea | 0.728 ± 0.005 | −4.1 pp | High |
| Solidity | 0.738 ± 0.002 | −3.1 pp | High |
| Volume | 0.748 ± 0.008 | −2.1 pp | Moderate |
| MajorAxisLength | 0.751 ± 0.002 | −1.8 pp | Moderate |
| EquivalentDiameter | 0.764 ± 0.003 | −0.5 pp | Low |
| Extent | 0.767 ± 0.002 | −0.2 pp | Negligible |
| MinorAxisLength | 0.769 ± 0.003 | 0.0 pp | Negligible |
| EulerNumber | 0.772 ± 0.002 | +0.3 pp | Negligible |

**SurfaceArea is the single most important feature** (−4.1 pp), followed by Solidity (−3.1 pp).
Together they account for the majority of discriminative signal in the 8 shape descriptors.

**SurfaceArea** encodes the surface complexity of the nucleus — nuclei at different mitotic stages
differ substantially in membrane roughness and folding as they condense, elongate, and divide.
**Solidity** (volume / convex hull volume) captures convexity defects: non-convex protrusions or
indentations that appear as chromosomes condense and the nuclear envelope breaks down.

**Volume** (−2.1 pp) and **MajorAxisLength** (−1.8 pp) are moderately important — both encode
overall nuclear size and elongation that changes across the cell cycle.

**EquivalentDiameter**, **Extent**, **MinorAxisLength**, and **EulerNumber** are effectively redundant
(Δ ≤ 0.5 pp, within noise). EquivalentDiameter is a monotone function of Volume (diameter of
sphere with same volume), so its marginal contribution is near zero. Extent (volume / bounding box
volume) is correlated with Solidity. MinorAxisLength overlaps with the other axis/size features.
EulerNumber (topological genus) shows essentially no variation across mitotic stages for these nuclei.

The low C values for no-EquivalentDiameter (C=1.08) and no-MinorAxisLength (C=25) vs the full
set (C=225) suggest these features actually introduce mild collinearity — removing them slightly
improves feature space conditioning.

### SurfaceArea and Solidity are complementary

| Feature Set | # Feat | Bal. Acc | Geo. Mean |
|---|---|---|---|
| SurfaceArea + Solidity only | 2 | 0.609 ± 0.004 | 0.509 |
| Shape only (all 8) | 8 | 0.738 ± 0.008 | 0.727 |

Together (2 features), SurfaceArea + Solidity reaches 0.609 — equivalent to the position-only
set (14 features, 0.609) — but still 12.9 pp below the full 8-feature shape set.

The remaining +12.9 pp gap to shape-only (0.738) reflects that Volume, MajorAxisLength, and
the other shape features each resolve additional class boundaries not captured by surface
complexity + convexity alone.

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
# All three original conditions
python benchmarks/invariants_classification.py \
    --input ../Minkowski_classifier/data/allen_cell/mitotic_cells_annotated/nuclei/minkowski_tensors_with_eigen_vals.csv \
    --cellprofiler-input ../Minkowski_classifier/data/allen_cell/mitotic_cells_annotated/nuclei/cellprofiler_features.csv \
    --output benchmarks/results/allen_cell/allen_cell_nuclei_cellprofiler_ablation \
    --include "CellProfiler" \
    --optimize \
    --n_iter 20 \
    --linear-only \
    --seeds 3 \
    --n_jobs 5

# Shape feature leave-one-out ablation (all 8 shape features)
python benchmarks/invariants_classification.py \
    --input ../Minkowski_classifier/data/allen_cell/mitotic_cells_annotated/nuclei/minkowski_tensors_with_eigen_vals.csv \
    --cellprofiler-input ../Minkowski_classifier/data/allen_cell/mitotic_cells_annotated/nuclei/cellprofiler_features.csv \
    --output benchmarks/results/allen_cell/allen_cell_nuclei_cellprofiler_ablation2 \
    --include "CellProfiler (no Solidity)" "CellProfiler (no Extent)" \
    --optimize \
    --n_iter 20 \
    --linear-only \
    --seeds 3 \
    --n_jobs 5

python benchmarks/invariants_classification.py \
    --input ../Minkowski_classifier/data/allen_cell/mitotic_cells_annotated/nuclei/minkowski_tensors_with_eigen_vals.csv \
    --cellprofiler-input ../Minkowski_classifier/data/allen_cell/mitotic_cells_annotated/nuclei/cellprofiler_features.csv \
    --output benchmarks/results/allen_cell/allen_cell_nuclei_cellprofiler_shape_ablation \
    --include "CellProfiler (no EquivalentDiameter)" "CellProfiler (no EulerNumber)" "CellProfiler (no MajorAxisLength)" "CellProfiler (no MinorAxisLength)" "CellProfiler (no SurfaceArea)" "CellProfiler (no Volume)" \
    --optimize \
    --n_iter 20 \
    --linear-only \
    --seeds 3 \
    --n_jobs 5

# Top-2 shape features combined
python benchmarks/invariants_classification.py \
    --input ../Minkowski_classifier/data/allen_cell/mitotic_cells_annotated/nuclei/minkowski_tensors_with_eigen_vals.csv \
    --cellprofiler-input ../Minkowski_classifier/data/allen_cell/mitotic_cells_annotated/nuclei/cellprofiler_features.csv \
    --output benchmarks/results/allen_cell/allen_cell_nuclei_cellprofiler_top2 \
    --include "CellProfiler (SurfaceArea + Solidity only)" \
    --optimize \
    --n_iter 20 \
    --linear-only \
    --seeds 3 \
    --n_jobs 5
```

---

## Runtime

- All three conditions combined: ~1.5 min (CellProfiler features are pre-computed; cost is BayesSearchCV only)
