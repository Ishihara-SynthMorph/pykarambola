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

| Feature | Description |
|---------|-------------|
| Image_Count_ConvertImageToObjects | Object count per image |
| AreaShape_BoundingBoxMaximum_X/Y/Z | Bounding box upper corners |
| AreaShape_BoundingBoxMinimum_X/Y/Z | Bounding box lower corners |
| AreaShape_BoundingBoxVolume | Bounding box volume |
| AreaShape_Center_X/Y/Z | Object centroid |
| AreaShape_EquivalentDiameter | Sphere-equivalent diameter |
| AreaShape_EulerNumber | Topological genus |
| AreaShape_Extent | Volume / bounding box volume |
| AreaShape_MajorAxisLength | Length of major principal axis |
| AreaShape_MinorAxisLength | Length of minor principal axis |
| AreaShape_Solidity | Volume / convex hull volume |
| AreaShape_SurfaceArea | Surface area |
| AreaShape_Volume | Voxel volume |

---

## Result

| Feature Set | # Features | Balanced Accuracy | Geo. Mean | Best C | Best PCA |
|-------------|------------|-------------------|-----------|--------|----------|
| CellProfiler | 22 | 0.769 ± 0.003 | 0.761 ± 0.003 | 225 | 22 |

---

## Full Combined Ranking

| Rank | Feature Set | # Features | Balanced Accuracy | Geo. Mean | Best C | Best PCA |
|------|-------------|------------|-------------------|-----------|--------|----------|
| 1 | Baseline (w/ eigen) | 86 | 0.818 ± 0.004 | 0.815 ± 0.004 | 1.08 | 84 |
| 1 | SO3 Degree 2 + Eigenvalues | 57 | 0.817 ± 0.003 | 0.814 ± 0.003 | 980 | 53 |
| 3 | SO2 Degree 1 + Eigenvalues | 36 | 0.799 ± 0.005 | 0.795 ± 0.006 | 13.3 | 25 |
| 4 | SO3 Degree 1 + Eigenvalues | 26 | 0.793 ± 0.006 | 0.789 ± 0.007 | 1000 | 26 |
| 5 | SO2 Degree 2 + Eigenvalues | 112 | 0.787 ± 0.001 | 0.781 ± 0.001 | 225 | 111 |
| 6 | SO3 Degree 2 + SO2 z-scalars | 49 | 0.784 ± 0.007 | 0.778 ± 0.008 | 225 | 49 |
| 6 | SO3 Degree 2 | 39 | 0.783 ± 0.002 | 0.778 ± 0.002 | 225 | 39 |
| 8 | **CellProfiler** | **22** | **0.769 ± 0.003** | **0.761 ± 0.003** | **225** | **22** |
| 9 | SO2 Degree 2 | 94 | 0.757 ± 0.008 | 0.751 ± 0.009 | 1000 | 94 |
| 10 | Baseline (tensors) | 62 | 0.746 ± 0.006 | 0.737 ± 0.006 | 1000 | 54 |
| 11 | SPHARM Inv lmax=5 | 75 | 0.726 ± 0.004 | 0.713 ± 0.006 | 739 | 57 |
| 12 | SO2 Degree 1 | 18 | 0.674 ± 0.006 | 0.649 ± 0.009 | 21.8 | 16 |
| 13 | SO3 Degree 1 | 8 | 0.667 ± 0.004 | 0.636 ± 0.005 | 995 | 8 |
| 14 | SPHARM lmax=5 | 72 | 0.597 ± 0.003 | 0.584 ± 0.004 | 0.39 | 57 |

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

### Runtime

CellProfiler features are pre-computed — feature extraction takes 0s (CSV join only).
Total benchmark runtime: 12 seconds.

---

## Configuration

```bash
python benchmarks/invariants_classification.py \
    --input ../Minkowski_classifier/data/allen_cell/mitotic_cells_annotated/nuclei/minkowski_tensors_with_eigen_vals.csv \
    --cellprofiler-input ../Minkowski_classifier/data/allen_cell/mitotic_cells_annotated/nuclei/cellprofiler_features.csv \
    --output benchmarks/results/allen_cell_nuclei_cellprofiler_optimized \
    --include "CellProfiler" \
    --optimize \
    --n_iter 20 \
    --linear-only \
    --seeds 3 \
    --n_jobs 5
```
