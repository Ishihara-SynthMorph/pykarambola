# Allen Cell Nuclei: SO3 Degree 3 + Eigenvalues — Optimized Results

## Overview

Bayesian-optimized classification result for SO3 Degree 3 invariants combined with
eigenvalues of rank-2 tensors, directly comparable to all previous optimized runs.

**Classifier**: LinearSVC (liblinear)
**Dataset**: Non-rotated nuclei only (`nuclei/`)
**Optimization**: BayesSearchCV, n_iter=20, 5-fold stratified CV
**Evaluation**: 3 seeds, balanced accuracy + geometric mean

---

## Result

| Feature Set | # Features | Balanced Accuracy | Geo. Mean | Best C | Best PCA |
|-------------|------------|-------------------|-----------|--------|----------|
| SO3 Degree 3 + Eigenvalues | 237 | 0.804 ± 0.005 | 0.797 ± 0.005 | 988 | 213 |

---

## Full Combined Ranking

| Rank | Feature Set | # Features | Balanced Accuracy | Geo. Mean | Best C | Best PCA |
|------|-------------|------------|-------------------|-----------|--------|----------|
| 1 | Baseline (w/ eigen) | 86 | 0.818 ± 0.004 | 0.815 ± 0.004 | 1.08 | 84 |
| 1 | SO3 Degree 2 + Eigenvalues | 57 | 0.817 ± 0.003 | 0.814 ± 0.003 | 980 | 53 |
| 3 | **SO3 Degree 3 + Eigenvalues** | **237** | **0.804 ± 0.005** | **0.797 ± 0.005** | **988** | **213** |
| 4 | SO2 Degree 1 + Eigenvalues | 36 | 0.799 ± 0.005 | 0.795 ± 0.006 | 13.3 | 25 |
| 5 | SO3 Degree 3 | 219 | 0.795 ± 0.004 | 0.786 ± 0.005 | 14.9 | 195 |
| 6 | SO3 Degree 1 + Eigenvalues | 26 | 0.793 ± 0.006 | 0.789 ± 0.007 | 1000 | 26 |
| 6 | Eigenvalues only | 18 | 0.791 ± 0.003 | 0.784 ± 0.005 | 37.2 | 16 |
| 8 | SO2 Degree 2 + Eigenvalues | 112 | 0.787 ± 0.001 | 0.781 ± 0.001 | 225 | 111 |
| 9 | SO3 Degree 2 + SO2 z-scalars | 49 | 0.784 ± 0.007 | 0.778 ± 0.008 | 225 | 49 |
| 9 | SO3 Degree 2 | 39 | 0.783 ± 0.002 | 0.778 ± 0.002 | 225 | 39 |
| 11 | CellProfiler | 22 | 0.769 ± 0.003 | 0.761 ± 0.003 | 225 | 22 |
| 12 | SO2 Degree 2 | 94 | 0.757 ± 0.008 | 0.751 ± 0.009 | 1000 | 94 |
| 13 | Baseline (tensors) | 62 | 0.746 ± 0.006 | 0.737 ± 0.006 | 1000 | 54 |
| 14 | SPHARM Inv lmax=5 | 75 | 0.726 ± 0.004 | 0.713 ± 0.006 | 739 | 57 |
| 15 | SO2 Degree 1 | 18 | 0.674 ± 0.006 | 0.649 ± 0.009 | 21.8 | 16 |
| 16 | SO3 Degree 1 | 8 | 0.667 ± 0.004 | 0.636 ± 0.005 | 995 | 8 |
| 17 | SPHARM lmax=5 | 72 | 0.597 ± 0.003 | 0.584 ± 0.004 | 0.39 | 57 |

---

## Interpretation

### Adding cubic invariants to eigenvalues hurts relative to degree 2

SO3 Degree 3 + Eigenvalues (0.804) is 1.3 pp *below* SO3 Degree 2 + Eigenvalues (0.817)
despite adding 180 cubic invariant features (57 → 237).
The degree-3 cross-tensor terms (Tr(WᵢWⱼWₖ)) add dimensionality without proportional
signal, diluting the eigenvalue subspace even after aggressive PCA (213/237).
The sweet spot for combining polynomial invariants with eigenvalues is degree 2:
quadratic cross-tensor products Tr(WᵢWⱼ) complement eigenvalues (+2.6 pp), but cubic
terms do not.

### C flips from 14.9 to 988 when eigenvalues are added to SO3 Degree 3

SO3 Degree 3 alone requires C=14.9 — the 219 cubic invariants are heavily collinear and
the classifier needs strong regularisation to suppress redundant directions.
Adding 18 eigenvalues shifts the optimal C to 988: eigenvalues provide clean, direct
representations of I₁/I₂/I₃ per tensor, and PCA (213/237) finds a subspace dominated
by the eigenvalue dimensions where features are more independent.
The same C flip was seen for SO3 Degree 2 (C=225) → SO3 Degree 2 + Eigenvalues (C=980),
confirming that eigenvalues systematically resolve the collinearity introduced by
polynomial invariants.

### Eigenvalue gain diminishes as base invariant degree increases

Across all SO3 + Eigenvalues combinations:

| Base | Base score | + Eigenvalues | Gain |
|------|------------|---------------|------|
| SO3 Degree 1 | 0.667 | 0.793 | +12.6 pp |
| SO3 Degree 2 | 0.783 | 0.817 | +3.4 pp |
| SO3 Degree 3 | 0.795 | 0.804 | +0.9 pp |

The gain shrinks at each degree because higher-degree invariants already recover more
of the eigenvalue information: degree 2 provides I₁ and I₂, degree 3 provides I₁, I₂,
and I₃ (via Tr(W³)). By degree 3, eigenvalues add only a cleaner/more direct
parameterisation of information the polynomial basis already encodes — a useful but
diminishing advantage.

### SO3 Degree 2 + Eigenvalues remains the optimal combination

At 57 features and 0.817, SO3 Degree 2 + Eigenvalues outperforms every larger feature
set tested, including SO3 Degree 3 + Eigenvalues (237 features, 0.804) and Baseline
(w/ eigen) (86 features, 0.818, statistically tied).
Increasing polynomial degree beyond 2 while eigenvalues are present adds noise faster
than signal for a linear classifier on this dataset.

---

## Runtime

- Total: ~35 min (n_iter=20, 1 feature set, 5-fold CV, 3 seeds, LinearSVC, n_jobs=5, Apple Silicon)

---

## Configuration

```bash
python benchmarks/invariants_classification.py \
    --input ../Minkowski_classifier/data/allen_cell/mitotic_cells_annotated/nuclei/minkowski_tensors_with_eigen_vals.csv \
    --output benchmarks/results/allen_cell_nuclei_so3d3_eigen_optimized \
    --include "SO3 Degree 3 + Eigenvalues" \
    --optimize \
    --n_iter 20 \
    --linear-only \
    --seeds 3 \
    --n_jobs 5
```
