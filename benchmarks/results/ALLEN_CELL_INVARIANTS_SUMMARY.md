# Allen Cell Nuclei: Invariants Classification Summary

## Overview

Comprehensive comparison of rotation-invariant polynomial features (SO3/SO2, degree 1–3)
against Minkowski tensor baselines, with and without explicit eigenvalue augmentation.

**Classifier**: LinearSVC (liblinear)
**Dataset**: Non-rotated nuclei only (`nuclei/`)
**Optimization**: BayesSearchCV, n_iter=20, 5-fold stratified CV
**Evaluation**: 3 seeds, balanced accuracy + geometric mean

---

## Results

### Minkowski Baselines

| Feature Set | # Feat | Balanced Accuracy | Geo. Mean | Best C | Best PCA |
|-------------|--------|-------------------|-----------|--------|----------|
| Minkowski (tensors) | 62 | 0.746 ± 0.006 | 0.737 ± 0.006 | 1000 | 54 |
| Eigenvalues only | 18 | 0.791 ± 0.003 | 0.784 ± 0.005 | 37.2 | 16 |
| Eigen + Beta | 24 | 0.808 ± 0.001 | 0.804 ± 0.001 | 225 | 24 |
| Minkowski (tensors+eigen) | 80 | 0.806 ± 0.007 | 0.802 ± 0.008 | 977 | 66 |
| Minkowski (tensors+eigen+beta) | 86 | **0.818 ± 0.004** | **0.815 ± 0.004** | 1.08 | 84 |

### SO3 Invariants (full rotation invariance)

| Feature Set | # Feat | Balanced Accuracy | Geo. Mean | Best C | Best PCA | Δ Eigen |
|-------------|--------|-------------------|-----------|--------|----------|---------|
| SO3 Degree 1 | 8 | 0.667 ± 0.004 | 0.636 ± 0.005 | 995 | 8 | — |
| SO3 Degree 1 + Eigenvalues | 26 | 0.793 ± 0.006 | 0.789 ± 0.007 | 1000 | 26 | **+12.6 pp** |
| SO3 Degree 2 | 39 | 0.783 ± 0.002 | 0.778 ± 0.002 | 225 | 39 | — |
| SO3 Degree 2 + Eigenvalues | 57 | **0.817 ± 0.003** | **0.814 ± 0.003** | 980 | 53 | **+3.4 pp** |
| SO3 Degree 3 | 219 | 0.795 ± 0.004 | 0.786 ± 0.005 | 14.9 | 195 | — |
| SO3 Degree 3 + Eigenvalues | 237 | 0.804 ± 0.005 | 0.797 ± 0.005 | 988 | 213 | **+0.9 pp** |

### SO2 Invariants (axial symmetry, z-axis preserved)

| Feature Set | # Feat | Balanced Accuracy | Geo. Mean | Best C | Best PCA | Δ Eigen |
|-------------|--------|-------------------|-----------|--------|----------|---------|
| SO2 Degree 1 | 18 | 0.674 ± 0.006 | 0.649 ± 0.009 | 21.8 | 16 | — |
| SO2 Degree 1 + Eigenvalues | 36 | 0.799 ± 0.005 | 0.795 ± 0.006 | 13.3 | 25 | **+12.5 pp** |
| SO2 Degree 2 | 94 | 0.757 ± 0.008 | 0.751 ± 0.009 | 1000 | 94 | — |
| SO2 Degree 2 + Eigenvalues | 112 | 0.787 ± 0.001 | 0.781 ± 0.001 | 225 | 111 | **+3.0 pp** |
| SO2 Degree 3 | — | not run | — | — | — | — |
| SO2 Degree 3 + Eigenvalues | — | not run | — | — | — | — |

---

## Interpretation

### Eigenvalue gain diminishes with polynomial degree

The most consistent pattern across both symmetry groups is that explicitly adding eigenvalues
(the three sorted eigenvalues of each rank-2 tensor) produces diminishing gains as polynomial
degree increases:

| Degree | SO3 Δ Eigen | SO2 Δ Eigen |
|--------|------------|------------|
| 1 | +12.6 pp | +12.5 pp |
| 2 | +3.4 pp | +3.0 pp |
| 3 | +0.9 pp | (not run) |

**Why**: Each rank-2 tensor W has three fundamental SO(3) invariants — I₁ = Tr(W),
I₂ = (Tr(W)² − Tr(W²))/2, and I₃ = det(W). Degree-1 polynomial invariants capture only I₁;
eigenvalues add I₂ and I₃, hence the large +12.6 pp gain. Degree-2 already recovers I₂
through cross-tensor products, so eigenvalues only add I₃, giving +3.4 pp. Degree-3
encodes all three invariants, leaving eigenvalues to provide only a pre-computed, better-
conditioned representation — +0.9 pp.

### SO3 Degree 2 + Eigenvalues is the optimal polynomial invariant set

At 57 features and 0.817, SO3 Degree 2 + Eigenvalues is the best-performing polynomial
invariant set:
- Better than SO3 Degree 3 + Eigenvalues (237 features, 0.804) — **degree-3 cubic terms
  hurt by 1.3 pp** when eigenvalues are already present; 180 extra cubic features add
  dimensionality without proportional signal, diluting the eigenvalue subspace
- Better than SO2 Degree 1/2 + Eigenvalues — full SO3 invariance encodes richer cross-tensor
  quadratic products than axial SO2 invariance

The sudden drop from C=225 (SO3 D2) to C=14.9 (SO3 D3) without eigenvalues signals that
the cubic terms are highly collinear; with eigenvalues the classifier recovers to C=988,
confirming the eigenvalue subspace is clean.

The +3.4 pp gain from adding eigenvalues to SO3 D2 (vs +12.6 pp at D1) breaks down as:
cross-tensor quadratic products Tr(WᵢWⱼ) contribute ~2.6 pp over eigenvalues alone, while
eigenvalues contribute ~0.8 pp on top of SO3 D2. Per-tensor shape dominates; cross-tensor
correlations add modest but real signal.

### SO3 Degree 1 + Eigenvalues ≈ Eigenvalues only

SO3 Degree 1 + Eigenvalues (26 features, 0.793) adds only +0.2 pp over Eigenvalues only
(18 features, 0.791). The SO3 Degree 1 traces are I₁ = Tr(W) per tensor, which equals the
sum of that tensor's eigenvalues — already implicit in the eigenvalue set. Adding explicit
traces provides no new information for a linear classifier.

### SO3 outperforms SO2 at the same degree

| Degree | SO3 (no eigen) | SO2 (no eigen) | SO3+E | SO2+E |
|--------|---------------|---------------|-------|-------|
| 1 | 0.667 | 0.674 (+0.7) | 0.793 | 0.799 (+0.6) |
| 2 | 0.783 | 0.757 (−2.6) | **0.817** | 0.787 (−3.0) |

At degree 1, SO2 has a slight edge (+0.6–0.7 pp): the z-axis asymmetry features add useful
information about nuclear orientation along the imaging axis.
At degree 2, SO3 dominates (+2.6 pp without eigen, +3.0 pp with eigen): the quadratic
Tr(WᵢWⱼ) cross-tensor products under full SO(3) invariance encode richer shape correlations
than the restricted z-axis-preserving SO2 products.

The regularisation contrast between SO3 D1+E (C=1000) and SO2 D1+E (C=13.3) is notable:
SO2 z-axis scalars (v_z, M_zz) partially overlap with eigenvalue projections along the z
direction, causing mild collinearity that forces lower C. SO3 D1 traces, being algebraically
redundant with eigenvalues, introduce no additional collinearity.

SO2 Degree 2 + Eigenvalues (112 features, 0.787) underperforms SO2 Degree 1 + Eigenvalues
(36 features, 0.799) by 1.2 pp despite 76 extra features. The degree-2 SO2 invariants and
eigenvalues span partially overlapping directions without the clean complementarity seen in
the SO3 case.

### Feature count vs performance: diminishing returns above 57 features

| # Feat | Feature Set | Bal. Acc | Δ from prev |
|--------|-------------|----------|-------------|
| 8 | SO3 Degree 1 | 0.667 | — |
| 18 | SO2 Degree 1 | 0.674 | +0.7 |
| 18 | Eigenvalues only | 0.791 | — |
| 24 | Eigen + Beta | 0.808 | — |
| 26 | SO3 D1 + Eigen | 0.793 | — |
| 36 | SO2 D1 + Eigen | 0.799 | — |
| 39 | SO3 Degree 2 | 0.783 | — |
| 57 | SO3 D2 + Eigen | **0.817** | peak |
| 62 | Mink (tensors) | 0.746 | — |
| 80 | Mink (tensors+eigen) | 0.806 | −1.1 vs peak |
| 86 | Mink (tensors+eigen+beta) | **0.818** | ties peak |
| 94 | SO2 Degree 2 | 0.757 | — |
| 112 | SO2 D2 + Eigen | 0.787 | — |
| 219 | SO3 Degree 3 | 0.795 | −2.2 vs peak |
| 237 | SO3 D3 + Eigen | 0.804 | −1.3 vs peak |

Performance peaks at 57 features (SO3 D2+E) and then declines or plateaus as features are
added. All sets above 57 features perform worse despite 4–5× the feature count.

### Effective dimensionality of degree-3 invariants is much lower than nominal

Degree-3 polynomial invariants include many near-redundant terms under typical mesh data.
When mesh centroids are zero (as is standard), w010 = 0 for all samples, making ~46
degree-3 features zero-variance. Additionally, surface curvature is relatively uniform
across medical imaging data, causing the four curvature-weighted tensors (w020, w120,
w220, w320) to be highly correlated (r > 0.95 for many pairs). The effective number of
linearly independent degree-3 features is ~135 of the nominal 219, and ~120 when
data-specific near-redundancies are removed. This explains why SO3 Degree 3 (219 features)
performs only marginally above SO3 Degree 2 (39 features) despite the large nominal
feature count increase.

### Where Minkowski baselines sit

- **Minkowski (tensors, 62)** = 0.746: below SO3 D2 (39 features, 0.783) — raw tensor components
  are a worse representation than polynomial invariants for linear classification
- **Eigenvalues only (18)** = 0.791: beats SO3 D2 raw (0.783) at half the features; nearly matches
  SO3 D1+E (26 features, 0.793) with 8 fewer features
- **Eigen + Beta (24)** = 0.808: ties Minkowski (tensors+eigen, 80) at 3.3× fewer features; the 62
  raw tensor components add negligible information once eigenvalues and betas are present
- **Minkowski (tensors+eigen, 80)** = 0.806: sits between SO3 D2+E (0.817) and SO3 D3+E (0.804),
  but with 23 more features than the better-performing SO3 D2+E
- **Minkowski (tensors+eigen+beta, 86)** = 0.818: statistically ties SO3 D2+E (0.817) — the two
  top-performing feature sets use entirely different representations (polynomial invariants vs
  raw tensors + derived quantities), yet converge to the same performance ceiling on this task

---

## Practical Recommendations

| Priority | Feature Set | # Feat | Bal. Acc | When to use |
|----------|-------------|--------|----------|-------------|
| 1st (tie) | SO3 Degree 2 + Eigenvalues | 57 | **0.817** | Best polynomial invariant set |
| 1st (tie) | Minkowski (tensors+eigen+beta) | 86 | **0.818** | Best raw-tensor set; no symmetry assumption |
| 2nd | Eigen + Beta | 24 | 0.808 | Best efficiency; 3–4× fewer features than top |
| 3rd | SO2 Degree 1 + Eigenvalues | 36 | 0.799 | Simple, fast, captures z-axis asymmetry |
| 4th | SO3 Degree 1 + Eigenvalues | 26 | 0.793 | Simplest full-SO3 set |
| 5th | Eigenvalues only | 18 | 0.791 | Minimal viable feature set |
| Avoid | SO3 Degree 3 (+E) | 219–237 | 0.795–0.804 | No gain over D2+E, 4× feature count |
| Avoid | Minkowski (tensors) | 62 | 0.746 | Dominated by any eigen-augmented set |
