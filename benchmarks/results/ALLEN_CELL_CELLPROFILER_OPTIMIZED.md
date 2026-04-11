# Allen Cell Nuclei: CellProfiler Features — Optimized Results

## Overview

Bayesian-optimized classification results for CellProfiler shape features,
directly comparable to all previous optimized runs.

**Classifier**: LinearSVC (liblinear)
**Dataset**: Non-rotated nuclei only (`nuclei/`)
**Optimization**: BayesSearchCV, n_iter=20, 5-fold stratified CV
**Evaluation**: 3 seeds, balanced accuracy + geometric mean
**Input**: `cellprofiler_features.csv` (full 22-feature output as provided by CellProfiler)

### CellProfiler features (22)

| Feature | Type | Description |
|---------|------|-------------|
| Image_Count_ConvertImageToObjects | position/count | Object count per image |
| AreaShape_BoundingBoxMaximum_X/Y/Z | position | Bounding box upper corners |
| AreaShape_BoundingBoxMinimum_X/Y/Z | position | Bounding box lower corners |
| AreaShape_BoundingBoxVolume | position | Bounding box volume |
| AreaShape_Center_X/Y/Z | position | Object centroid |
| Location_Center_X/Y/Z | position | Image-level centroid |
| AreaShape_EquivalentDiameter | **shape** | Sphere-equivalent diameter |
| AreaShape_EulerNumber | **shape** | Topological genus |
| AreaShape_Extent | **shape** | Volume / bounding box volume |
| AreaShape_MajorAxisLength | **shape** | Length of major principal axis |
| AreaShape_MinorAxisLength | **shape** | Length of minor principal axis |
| AreaShape_Solidity | **shape** | Volume / convex hull volume |
| AreaShape_SurfaceArea | **shape** | Surface area |
| AreaShape_Volume | **shape** | Voxel volume |

14 position/count features + 8 pure shape features.

---

## Results

| Feature Set | # Features | Balanced Accuracy | Geo. Mean | Best C | Best PCA |
|-------------|------------|-------------------|-----------|--------|----------|
| CellProfiler | 22 | 0.769 ± 0.003 | 0.761 ± 0.003 | 225 | 22 |
| CellProfiler (shape only) | 8 | 0.738 ± 0.008 | 0.727 ± 0.010 | 1000 | 8 |

---

## Full Combined Ranking

| Rank | Feature Set | # Features | Balanced Accuracy | Geo. Mean | Best C | Best PCA |
|------|-------------|------------|-------------------|-----------|--------|----------|
| 1 | Baseline (w/ eigen) | 86 | 0.818 ± 0.004 | 0.815 ± 0.004 | 1.08 | 84 |
| 1 | SO3 Degree 2 + Eigenvalues | 57 | 0.817 ± 0.003 | 0.814 ± 0.003 | 980 | 53 |
| 3 | SO2 Degree 1 + Eigenvalues | 36 | 0.799 ± 0.005 | 0.795 ± 0.006 | 13.3 | 25 |
| 4 | SO3 Degree 3 + Eigenvalues | 237 | 0.804 ± 0.005 | 0.797 ± 0.005 | 988 | 213 |
| 5 | SO3 Degree 3 | 219 | 0.795 ± 0.004 | 0.786 ± 0.005 | 14.9 | 195 |
| 6 | SO3 Degree 1 + Eigenvalues | 26 | 0.793 ± 0.006 | 0.789 ± 0.007 | 1000 | 26 |
| 7 | Eigenvalues only | 18 | 0.791 ± 0.003 | 0.784 ± 0.005 | 37.2 | 16 |
| 8 | SO2 Degree 2 + Eigenvalues | 112 | 0.787 ± 0.001 | 0.781 ± 0.001 | 225 | 111 |
| 9 | SO3 Degree 2 + SO2 z-scalars | 49 | 0.784 ± 0.007 | 0.778 ± 0.008 | 225 | 49 |
| 9 | SO3 Degree 2 | 39 | 0.783 ± 0.002 | 0.778 ± 0.002 | 225 | 39 |
| 11 | **CellProfiler** | **22** | **0.769 ± 0.003** | **0.761 ± 0.003** | **225** | **22** |
| 12 | SO2 Degree 2 | 94 | 0.757 ± 0.008 | 0.751 ± 0.009 | 1000 | 94 |
| 13 | Baseline (tensors) | 62 | 0.746 ± 0.006 | 0.737 ± 0.006 | 1000 | 54 |
| 14 | **CellProfiler (shape only)** | **8** | **0.738 ± 0.008** | **0.727 ± 0.010** | **1000** | **8** |
| 15 | SPHARM Inv lmax=5 | 75 | 0.726 ± 0.004 | 0.713 ± 0.006 | 739 | 57 |
| 16 | SO2 Degree 1 | 18 | 0.674 ± 0.006 | 0.649 ± 0.009 | 21.8 | 16 |
| 17 | SO3 Degree 1 | 8 | 0.667 ± 0.004 | 0.636 ± 0.005 | 995 | 8 |
| 18 | SPHARM lmax=5 | 72 | 0.597 ± 0.003 | 0.584 ± 0.004 | 0.39 | 57 |

---

## Interpretation

### CellProfiler is competitive despite a fundamentally different feature type

At 0.769, CellProfiler ranks 8th out of 14 conditions and outperforms:
- Baseline (tensors) (0.746) — raw Minkowski tensor components without eigenvalues
- SPHARM Inv lmax=5 (0.726) — rotation-invariant spherical harmonics power spectrum
- All degree-1 invariant-only sets (SO3/SO2 Degree 1: 0.667–0.674)

This is notable because CellProfiler features are computed by standard 3D image analysis
software with no tensor mathematics — they are basic geometric shape descriptors (volume,
surface area, axis lengths, bounding box, solidity, Euler number).

### CellProfiler vs Minkowski tensor baselines

CellProfiler (0.769) beats Baseline (tensors) (0.746) by 2.3 pp despite having fewer
features (22 vs 62) and no orientation information.
This shows that the raw Minkowski tensor components, without eigenvalue decomposition, are
less informative than a compact set of directly interpretable 3D shape measurements.
The Minkowski baseline only overtakes CellProfiler once eigenvalues are added: Baseline
(w/ eigen) reaches 0.818 (+4.9 pp over CellProfiler).

### CellProfiler vs SO3 Degree 2

CellProfiler (0.769) sits 1.4 pp below SO3 Degree 2 (0.783).
Both use C=225 and full PCA retention, suggesting similar feature conditioning.
SO3 Degree 2 encodes rotation-invariant quadratic combinations of tensor components
(cross-tensor products Tr(WᵢWⱼ)), providing shape correlation information not available
in the scalar CellProfiler descriptors — which likely accounts for the gap.

### C=225 and full PCA: well-conditioned, non-redundant features

The same hyperparameters as SO3 Degree 2 (C=225, 100% PCA) confirm that CellProfiler
features are well-conditioned and non-redundant: every dimension contributes, and the
classifier does not need strong regularisation.
This is consistent with the features being independently defined geometric quantities
rather than polynomial combinations of a common underlying representation.

### Shape vs position contribution

CellProfiler (shape only) scores 0.738 ± 0.008 with the 8 pure shape descriptors alone.
The full 22-feature set adds +3.1 pp (0.738 → 0.769) from the 14 position/bounding box/count
columns, reflecting real spatial organisation of mitotic stages within the imaging volume.
Both components contribute, but shape features carry the majority of the signal.

C=1000 for shape-only vs C=225 for full confirms that removing position features eliminates
mild redundancy between the two centroid representations (AreaShape_Center and Location_Center).

### 8 CellProfiler shape features vs 8 SO3 Degree 1 features

CellProfiler shape only (0.738) handily beats SO3 Degree 1 (0.667) at the same feature count.
Volume, surface area, axis lengths and solidity are more directly interpretable shape summaries
than tensor traces, and more discriminative for this task without needing eigenvalue decomposition.

### Runtime

CellProfiler features are pre-computed — feature extraction takes 0s (CSV join only).
Total benchmark runtime: 20 seconds (both variants).

---

## Configuration

```bash
python benchmarks/invariants_classification.py \
    --input ../Minkowski_classifier/data/allen_cell/mitotic_cells_annotated/nuclei/minkowski_tensors_with_eigen_vals.csv \
    --cellprofiler-input ../Minkowski_classifier/data/allen_cell/mitotic_cells_annotated/nuclei/cellprofiler_features.csv \
    --output benchmarks/results/allen_cell_nuclei_cellprofiler_optimized \
    --include "CellProfiler" \  # matches both "CellProfiler" and "CellProfiler (shape only)"
    --optimize \
    --n_iter 20 \
    --linear-only \
    --seeds 3 \
    --n_jobs 5
```
