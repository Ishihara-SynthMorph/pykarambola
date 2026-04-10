# Allen Cell Nuclei: Eigenvalue Combination Feature Sets — Optimized Results

## Overview

Bayesian-optimized results for three eigenvalue combination feature sets, directly comparable
to all previous optimized runs.
The goal is to isolate the eigenvalue contribution at each invariant degree and symmetry.

**Classifier**: LinearSVC (liblinear)
**Dataset**: Non-rotated nuclei only (`nuclei/`)
**Optimization**: BayesSearchCV, n_iter=20, 5-fold stratified CV
**Evaluation**: 3 seeds, balanced accuracy + geometric mean

---

## New Results

| Feature Set | # Features | Balanced Accuracy | Geo. Mean | Best C | Best PCA |
|-------------|------------|-------------------|-----------|--------|----------|
| SO3 Degree 1 + Eigenvalues | 26 | 0.793 ± 0.006 | 0.789 ± 0.007 | 1000 | 26 |
| SO2 Degree 1 + Eigenvalues | 36 | **0.799 ± 0.005** | **0.795 ± 0.006** | 13.3 | 25 |
| SO2 Degree 2 + Eigenvalues | 112 | 0.787 ± 0.001 | 0.781 ± 0.001 | 225 | 111 |

---

## Full Combined Ranking

| Rank | Feature Set | # Features | Balanced Accuracy | Geo. Mean | Best C | Best PCA |
|------|-------------|------------|-------------------|-----------|--------|----------|
| 1 | Baseline (w/ eigen) | 86 | 0.818 ± 0.004 | 0.815 ± 0.004 | 1.08 | 84 |
| 1 | SO3 Degree 2 + Eigenvalues | 57 | 0.817 ± 0.003 | 0.814 ± 0.003 | 980 | 53 |
| 3 | **SO2 Degree 1 + Eigenvalues** | **36** | **0.799 ± 0.005** | **0.795 ± 0.006** | **13.3** | **25** |
| 4 | **SO3 Degree 1 + Eigenvalues** | **26** | **0.793 ± 0.006** | **0.789 ± 0.007** | **1000** | **26** |
| 5 | **SO2 Degree 2 + Eigenvalues** | **112** | **0.787 ± 0.001** | **0.781 ± 0.001** | **225** | **111** |
| 6 | SO3 Degree 2 + SO2 z-scalars | 49 | 0.784 ± 0.007 | 0.778 ± 0.008 | 225 | 49 |
| 6 | SO3 Degree 2 | 39 | 0.783 ± 0.002 | 0.778 ± 0.002 | 225 | 39 |
| 8 | SO2 Degree 2 | 94 | 0.757 ± 0.008 | 0.751 ± 0.009 | 1000 | 94 |
| 9 | Baseline (tensors) | 62 | 0.746 ± 0.006 | 0.737 ± 0.006 | 1000 | 54 |
| 10 | SPHARM Inv lmax=5 | 75 | 0.726 ± 0.004 | 0.713 ± 0.006 | 739 | 57 |
| 11 | SO2 Degree 1 | 18 | 0.674 ± 0.006 | 0.649 ± 0.009 | 21.8 | 16 |
| 12 | SO3 Degree 1 | 8 | 0.667 ± 0.004 | 0.636 ± 0.005 | 995 | 8 |
| 13 | SPHARM lmax=5 | 72 | 0.597 ± 0.003 | 0.584 ± 0.004 | 0.39 | 57 |

---

## Interpretation

### Eigenvalue gain is inversely proportional to the degree of the base invariants

For a 3×3 symmetric tensor W, the complete per-tensor shape descriptor requires three invariants:
I₁ = Tr(W), I₂ = ½[(Tr W)² − Tr(W²)], I₃ = det(W) — equivalent to the sorted eigenvalues.

Degree-1 invariants capture only I₁; degree-2 also captures I₂.
Adding eigenvalues therefore fills a larger gap at lower degrees:

| Base invariants | What eigenvalues add | Gain |
|-----------------|----------------------|------|
| SO3 Degree 1 (I₁ only) | I₂ + I₃ | +12.6 pp (0.667 → 0.793) |
| SO2 Degree 1 (I₁ + z-scalars) | I₂ + I₃ | +12.5 pp (0.674 → 0.799) |
| SO2 Degree 2 (I₁ + I₂ + z-scalars) | I₃ only | +3.0 pp (0.757 → 0.787) |
| SO3 Degree 2 (I₁ + I₂ + cross-tensors) | I₃ only | +3.4 pp (0.783 → 0.817) |

The consistent ~3 pp gain when only I₃ is missing, and ~12–13 pp gain when both I₂ and I₃ are
missing, confirms that eigenvalues act as a compact complete per-tensor shape descriptor rather
than adding a single specific piece of information.

### SO3 Degree 1 + Eigenvalues (26 features) outperforms SO3 Degree 2 (39 features)

0.793 vs 0.783 — 26 features beat 39 with the same optimizer budget.
SO3 Degree 2 replaces the missing I₂ and I₃ with degree-2 polynomial combinations, but these
cross-tensor quadratic terms (Tr(WᵢWⱼ)) only partially recover per-tensor shape information.
The direct eigenvalue parameterisation gives the classifier the complete per-tensor spectrum
without having to reconstruct it from polynomial combinations.

### SO2 Degree 1 + Eigenvalues (36 features) reaches rank 3 overall

At 0.799, it sits only 1.8 pp below the best (0.817/0.818) at less than half the feature count.
The z-axis scalars in SO2 Degree 1 (v_z and M_zz components) provide a small additional edge
over SO3 Degree 1 + Eigenvalues (0.793), consistent with z-axis orientation being a weak but
real discriminant on non-rotated nuclei.
The low C (13.3) signals mild collinearity: some z-axis scalars (e.g. M_zz) partially overlap
with eigenvalue projections, requiring more regularisation than the clean SO3 case (C=1000).

### SO2 Degree 2 + Eigenvalues (112 features) underperforms relative to its component sizes

At 0.787, it is below both SO3 Degree 2 + Eigenvalues (0.817) and above SO2 Degree 2 alone
(0.757), but the 76 extra degree-2 SO2 features add noise rather than signal when eigenvalues
are already present.
The moderate C (225) and near-full PCA retention (111/112) confirm the optimizer could not find
a compact subspace — the degree-2 SO2 invariants and eigenvalues span overlapping directions
without forming a clean complementary pair.

### The eigenvalue I₂+I₃ combination is the primary driver of classification performance

Removing just these two pieces of per-tensor information (going from X+Eigenvalues to X alone)
costs 3–13 pp depending on whether the base already contains I₂.
The degree-2 cross-tensor polynomial terms in SO3 Degree 2 contribute only ~1 pp net over the
simpler SO3 Degree 1 + Eigenvalues (0.793 → 0.817 after also adding I₃), suggesting
cross-tensor quadratic correlations have modest discriminative value for this task.

---

## Runtime

- Total: ~38 min (n_iter=20, 3 feature sets, 5-fold CV, 3 seeds, LinearSVC, n_jobs=5, Apple Silicon)

---

## Configuration

```bash
python benchmarks/invariants_classification.py \
    --input ../Minkowski_classifier/data/allen_cell/mitotic_cells_annotated/nuclei/minkowski_tensors_with_eigen_vals.csv \
    --output benchmarks/results/allen_cell_nuclei_eigen_combinations_optimized \
    --max-so3-degree 1 \
    --max-so2-degree 2 \
    --include "SO3 Degree 1 + Eigenvalues" "SO2 Degree 1 + Eigenvalues" "SO2 Degree 2 + Eigenvalues" \
    --optimize \
    --n_iter 20 \
    --linear-only \
    --seeds 3 \
    --n_jobs 5
```
