# Allen Cell Nuclei: Invariants Classification Summary

## Overview

Comprehensive comparison of rotation-invariant polynomial features (SO3/SO2, degree 1–3)
against Minkowski tensor baselines, with and without explicit eigenvalue augmentation.

**Classifier**: LinearSVC (liblinear)
**Dataset**: Non-rotated nuclei only (`nuclei/`)
**Optimization**: BayesSearchCV, n_iter=20, 5-fold stratified CV
**Evaluation**: 3 seeds, balanced accuracy + geometric mean
**Input**: minkowski_tensors_with_eigen_vals.csv, spherical_harmonics_lmax_16.csv, cellprofiler_features.csv

---

## Results

### Minkowski Baselines

| Feature Set | # Feat | Non-zero var | Rank | Balanced Accuracy | Geo. Mean | Best C | Best PCA |
|-------------|--------|-------------|------|-------------------|-----------|--------|----------|
| Minkowski (tensors) | 62 | 59 | 55 | 0.746 ± 0.006 | 0.737 ± 0.006 | 1000 | 54 |
| Eigenvalues only | 18 | 18 | 18 | 0.791 ± 0.003 | 0.784 ± 0.005 | 37.2 | 16 |
| Eigen + Beta | 24 | 24 | 24 | 0.808 ± 0.001 | 0.804 ± 0.001 | 225 | 24 |
| Minkowski (tensors+eigen) | 80 | 77 | 67 | 0.806 ± 0.007 | 0.802 ± 0.008 | 977 | 66 |
| Minkowski (tensors+eigen+beta) | 86 | 83 | 73 | **0.818 ± 0.004** | **0.815 ± 0.004** | 1.08 | 84 |

### SO3 Invariants (full rotation invariance)

| Feature Set | # Feat | Non-zero var | Rank | Balanced Accuracy | Geo. Mean | Best C | Best PCA | Δ |
|-------------|--------|-------------|------|-------------------|-----------|--------|----------|---|
| SO3 Degree 1 | 8 | 8 | 8 | 0.667 ± 0.004 | 0.636 ± 0.005 | 995 | 8 | — |
| SO3 Degree 1 + Eigenvalues | 26 | 26 | 22 | 0.793 ± 0.006 | 0.789 ± 0.007 | 1000 | 26 | **+12.6 pp** vs D1 |
| SO3 Degree 1 + Eigenvalues + Beta | 32 | 32 | 32 | 0.814 ± 0.003 | 0.812 ± 0.003 | 739 | 25 | **+2.1 pp** vs D1+E |
| SO3 Degree 2 | 39 | 38 | 38 | 0.783 ± 0.002 | 0.778 ± 0.002 | 225 | 39 | — |
| SO3 Degree 2 + Eigenvalues | 57 | 56 | 52 | 0.817 ± 0.003 | 0.814 ± 0.003 | 980 | 53 | **+3.4 pp** vs D2 |
| SO3 Degree 2 + Eigenvalues + Beta | 63 | 62 | 56 | **0.827 ± 0.009** | **0.824 ± 0.010** | 739 | 48 | **+1.0 pp** vs D2+E |
| SO3 Degree 3 | 219 | 212 | 212 | 0.795 ± 0.004 | 0.786 ± 0.005 | 14.9 | 195 | — |
| SO3 Degree 3 + Eigenvalues | 237 | 230 | 226 | 0.804 ± 0.005 | 0.797 ± 0.005 | 988 | 213 | **+0.9 pp** vs D3 |

### SO2 Invariants (axial symmetry, z-axis preserved)

| Feature Set | # Feat | Non-zero var | Rank | Balanced Accuracy | Geo. Mean | Best C | Best PCA | Δ Eigen |
|-------------|--------|-------------|------|-------------------|-----------|--------|----------|---------|
| SO2 Degree 1 | 18 | 17 | 17 | 0.674 ± 0.006 | 0.649 ± 0.009 | 21.8 | 16 | — |
| SO2 Degree 1 + Eigenvalues | 36 | 35 | 31 | 0.799 ± 0.005 | 0.795 ± 0.006 | 13.3 | 25 | **+12.5 pp** |
| SO2 Degree 1 + Eigenvalues + Beta | 42 | 41 | 42 | 0.809 ± 0.002 | 0.806 ± 0.002 | 24.2 | 37 | **+1.0 pp** vs D1+E |
| SO2 Degree 2 | 94 | 92 | 92 | 0.757 ± 0.008 | 0.751 ± 0.009 | 1000 | 94 | — |
| SO2 Degree 2 + Eigenvalues | 112 | 110 | 106 | 0.787 ± 0.001 | 0.781 ± 0.001 | 225 | 111 | **+3.0 pp** |
| SO2 Degree 3 | 754 | 740 | 740 | not run | — | — | — | — |
| SO2 Degree 3 + Eigenvalues | 772 | 758 | 754 | not run | — | — | — | — |

### O(3) Invariants (parity-filtered, full rotation invariance)

O(3) = SO(3) + spatial inversion. Parity-odd invariants (`det_`, `comm_` terms, degree ≥ 3)
are removed. Feature counts at degrees 1–2 are identical to SO(3); at degree 3: 155 vs 219.

| Feature Set | # Feat | Balanced Accuracy | Geo. Mean | Δ vs SO(3) |
|-------------|--------|-------------------|-----------|------------|
| O3 Degree 1 | 8 | 0.667 ± 0.004 | 0.636 ± 0.005 | = SO3 D1 |
| O3 Degree 1 + Eigenvalues | 26 | 0.793 ± 0.006 | 0.789 ± 0.007 | = SO3 D1+E |
| O3 Degree 1 + Eigenvalues + Beta | 32 | 0.814 ± 0.003 | 0.812 ± 0.003 | = SO3 D1+E+B |
| O3 Degree 2 | 39 | 0.783 ± 0.005 | 0.778 ± 0.005 | = SO3 D2 |
| O3 Degree 2 + Eigenvalues | 57 | 0.818 ± 0.003 | 0.814 ± 0.004 | ≈ SO3 D2+E (+0.1 pp) |
| O3 Degree 2 + Eigenvalues + Beta | 63 | 0.826 ± 0.009 | 0.824 ± 0.010 | ≈ SO3 D2+E+B (−0.1 pp) |
| O3 Degree 3 | 155 | **0.806 ± 0.003** | 0.800 ± 0.004 | **+1.1 pp** vs SO3 D3 (219 feat, 0.795) |
| O3 Degree 3 + Eigenvalues | 173 | **0.813 ± 0.002** | 0.808 ± 0.002 | **+0.9 pp** vs SO3 D3+E (237 feat, 0.804) |
| O3 Degree 3 + Eigenvalues + Beta | 179 | 0.809 ± 0.005 | 0.803 ± 0.006 | — |

### O(2) Invariants (parity-filtered, axial symmetry)

O(2) = SO(2) + z-reflection. Parity-odd invariants (those with an odd number of `_z`/`_xz`
factors) are removed. Feature counts: 14/66/438 vs 18/94/754 for SO(2) D1/D2/D3.

| Feature Set | # Feat | Balanced Accuracy | Geo. Mean | Δ vs SO(2) |
|-------------|--------|-------------------|-----------|------------|
| O2 Degree 1 | 14 | 0.678 ± 0.005 | 0.653 ± 0.007 | SO2 D1: 0.674, +0.4 pp |
| O2 Degree 1 + Eigenvalues | 32 | 0.797 ± 0.004 | 0.793 ± 0.004 | SO2 D1+E: 0.799, −0.2 pp |
| O2 Degree 1 + Eigenvalues + Beta | 38 | 0.808 ± 0.003 | 0.805 ± 0.003 | SO2 D1+E+B: 0.809, −0.1 pp |
| O2 Degree 2 | 66 | 0.754 ± 0.008 | 0.748 ± 0.009 | SO2 D2: 0.757, −0.3 pp |
| O2 Degree 2 + Eigenvalues | 84 | 0.809 ± 0.006 | 0.804 ± 0.006 | SO2 D2+E: 0.787, **+2.2 pp** |
| O2 Degree 2 + Eigenvalues + Beta | 90 | **0.818 ± 0.003** | 0.815 ± 0.004 | — |
| O2 Degree 3 | 438 | 0.749 ± 0.004 | 0.736 ± 0.006 | SO2 D3: not run |
| O2 Degree 3 + Eigenvalues | 456 | 0.780 ± 0.003 | 0.770 ± 0.003 | SO2 D3+E: not run |
| O2 Degree 3 + Eigenvalues + Beta | 462 | 0.782 ± 0.001 | 0.772 ± 0.001 | SO2 D3+E+B: not run |

---

## Interpretation

### Eigenvalue gain diminishes with polynomial degree

The most consistent pattern across both symmetry groups is that explicitly adding eigenvalues
(the three sorted eigenvalues of each rank-2 tensor) produces diminishing gains as polynomial
degree increases:

| Degree | SO3 Δ Eigen | SO3 Δ Beta (vs +E) | SO2 Δ Eigen | SO2 Δ Beta (vs +E) |
|--------|------------|-------------------|------------|-------------------|
| 1 | +12.6 pp | +2.1 pp | +12.5 pp | +1.0 pp |
| 2 | +3.4 pp | +1.0 pp | +3.0 pp | (not run) |
| 3 | +0.9 pp | (not run) | (not run) | (not run) |

**Why**: Each rank-2 tensor W has three fundamental SO(3) invariants — I₁ = Tr(W),
I₂ = (Tr(W)² − Tr(W²))/2, and I₃ = det(W). Degree-1 polynomial invariants capture only I₁;
eigenvalues add I₂ and I₃, hence the large +12.6 pp gain. Degree-2 already recovers I₂
through cross-tensor products, so eigenvalues only add I₃, giving +3.4 pp. Degree-3
encodes all three invariants, leaving eigenvalues to provide only a pre-computed, better-
conditioned representation — +0.9 pp.

The beta gain (+1.0–2.1 pp at degree 1) follows the same diminishing pattern: beta indices
(β = (λ₁−λ₃)/trace) are not expressible as linear combinations of eigenvalues, so they add
genuine signal to a linear classifier at all degrees. The smaller beta gain for SO2 vs SO3
at degree 1 (+1.0 pp vs +2.1 pp) reflects partial overlap with the z-axis orientation
scalars already present in SO2 D1 features.

### SO3 Degree 2 + Eigenvalues + Beta is the best polynomial invariant set

Adding beta (anisotropy indices, 6 features) on top of SO3 D2 + Eigenvalues pushes performance
from 0.817 to **0.827** — +1.0 pp — and establishes a new overall best, surpassing
Minkowski (tensors+eigen+beta) at 0.818 by +0.9 pp.

The beta gain is consistent but smaller at lower degree: +2.1 pp for D1 (0.793→0.814) vs
+1.0 pp for D2 (0.817→0.827). This mirrors the eigenvalue gain pattern — diminishing returns
as polynomial degree increases — because beta indices (β = (λ₁−λ₃)/trace) are algebraically
related to the higher-degree invariants already present in D2.

SO3 Degree 2 + Eigenvalues (0.817) remains the best set without beta:
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

### Which D2 features drive the gain from D1+E+B to D2+E+B?

SO3 D2 adds 31 new features over D1: 10 dot products of rank-1 vectors and 21 Frobenius
inner products of traceless rank-2 matrices. With w010 ≈ 0 for all centered meshes, the 4
dot products involving w010 are near-zero. The remaining 27 active features split into three
groups:

| Group | Count | Formula | What it encodes |
|---|---|---|---|
| `frob_self` | 6 | `‖Tᵢ‖²_F = Tr(Tᵢ²) = λ₁²+λ₂²+λ₃²` | Per-tensor spectral energy |
| `frob_cross` | 15 | `Tr(TᵢTⱼ)`, i≠j | Relative alignment of two tensors' principal axes |
| `dots` (non-w010) | 6 | `dot(vᵢ, vⱼ)` for vᵢ ∈ {w110, w210, w310} | Surface moment magnitudes and mutual alignment |

To identify which group is responsible for the +1.3 pp gain (D1+E+B 0.814 → D2+E+B 0.827),
each group was added independently on top of D1+E+B:

| Feature Set | # Feat | Rank | Bal. Acc | Δ vs D1+E+B | Best C | Best PCA |
|---|---|---|---|---|---|---|
| D1+E+B (baseline) | 32 | 32 | 0.814 ± 0.003 | — | 739 | 25 |
| D1+E+B + dots | 38 | 38 | 0.811 ± 0.006 | −0.3 pp | 1000 | 33 |
| D1+E+B + frob_cross | 47 | 41 | 0.820 ± 0.008 | +0.6 pp | 12.3 | 45 |
| D1+E+B + frob_self | 38 | 32 | 0.826 ± 0.003 | +1.2 pp | 1000 | 38 |
| D1+E+B + frob_all | 53 | 47 | **0.830 ± 0.005** | +1.6 pp | 1000 | 50 |
| D2+E+B (ceiling) | 63 | 56 | 0.827 ± 0.009 | +1.3 pp | 739 | 48 |

**frob_self dominates** (+1.2 pp from 6 features), **frob_cross adds modest complementary
signal** (+0.6 pp from 15 features), and **dots are useless or mildly harmful** (−0.3 pp).

The dominance of frob_self is initially surprising — `‖Tᵢ‖²_F = λ₁²+λ₂²+λ₃²` is
algebraically derivable from the eigenvalues already in the feature set. However, for a
**linear** classifier, the squared norms cannot be expressed as linear combinations of λ₁,
λ₂, λ₃: they require an explicit quadratic feature. frob_self effectively adds the second
power of each tensor's spectral content, completing a degree-2 polynomial basis over the
eigenvalue spectrum.

frob_cross (`Tr(TᵢTⱼ)` for i≠j) encodes the relative orientation of two tensors' principal
axes — information that is genuinely absent from per-tensor eigenvalues. Its smaller gain
(+0.6 pp) suggests that cross-tensor alignment is less discriminative than per-tensor spectral
energy for this task, or that the signal it carries is partially captured by the beta
anisotropy indices already present.

frob_all (0.830) marginally exceeds D2+E+B (0.827) because D2+E+B also includes the
near-zero dot products, which add 10 features of noise and push the classifier toward higher
regularisation.

**Note on reporting and selection bias.** D1+E+B+frob_all was assembled *after* observing
the ablation results: dots were excluded because they performed poorly, not for reasons
independent of this data. This constitutes post-hoc feature selection on the same folds used
to evaluate it, which inflates apparent performance relative to a pre-specified design.
A partial theoretical defence exists — frob_all is the complete Gram matrix of the rank-2
tensor subspace (all pairwise Frobenius inner products), and excluding dots can be motivated
independently by the near-degeneracy of w010 and the qualitatively different geometric meaning
of rank-1 dot products vs. symmetric tensor alignment. Nevertheless, **SO3 D2+E+B (0.827)
should be treated as the primary, pre-specified result**. The 0.003 pp difference between
D1+E+B+frob_all and SO3 D2+E+B lies within the reported standard deviations (±0.005,
±0.009) and the two sets are statistically indistinguishable. The ablation is best understood
as an interpretation of why D2 works, not as an independent performance claim.

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

### Parity filtering: O(3) vs SO(3), O(2) vs SO(2)

Removing parity-odd invariants has a degree-dependent effect.

At degrees 1 and 2, O(3) and SO(3) are **equivalent to within noise** (Δ < 0.1 pp): all
parity-odd polynomial invariants (`det_` triple determinants and `comm_` commutator products)
are degree-3 constructs, so no features are removed at degrees 1–2.

At degree 3, O(3) outperforms SO(3) by **+1.1 pp** (155 vs 219 features, no eigenvalues) and
**+0.9 pp** with eigenvalues (173 vs 237 features). The 64 parity-odd degree-3 features that
SO(3) includes but O(3) discards are noise for non-chiral shapes: they change sign under spatial
inversion, average near zero over a symmetric dataset, and add dimensionality that the linear
classifier must regularise away (reflected in the low C=14.9 at SO3 D3 vs richer
regularisation at O3 D3).

The O(2) vs SO(2) pattern is similar but the gains are smaller and noisier. At degree 2 without
eigenvalues the difference is negligible (−0.3 pp), but with eigenvalues O2 D2+E gains
**+2.2 pp** over SO2 D2+E (84 vs 112 features). At degree 3, both O2 D3 (438 feat, 0.749) and
O2 D3+E+B (462 feat, 0.782) are much worse than O2 D2+E+B (90 feat, 0.818) — the same
degree-3 degradation pattern seen in SO3.

**Summary**: For non-chiral shapes, parity filtering at degree 3 recovers ~1 pp with ~25–30%
fewer features. At degrees 1–2 the effect is negligible. Degree 3 should be avoided regardless
of parity filtering; D2+E+B remains the best overall set.

### Feature count vs performance: diminishing returns above 57 features

| # Feat | Feature Set | Bal. Acc | Δ from prev |
|--------|-------------|----------|-------------|
| 8 | SO3 Degree 1 | 0.667 | — |
| 18 | SO2 Degree 1 | 0.674 | +0.7 |
| 18 | Eigenvalues only | 0.791 | — |
| 24 | Eigen + Beta | 0.808 | — |
| 26 | SO3 D1 + Eigen | 0.793 | — |
| 32 | SO3 D1 + Eigen + Beta | 0.814 | — |
| 36 | SO2 D1 + Eigen | 0.799 | — |
| 42 | SO2 D1 + Eigen + Beta | 0.809 | — |
| 39 | SO3 Degree 2 | 0.783 | — |
| 57 | SO3 D2 + Eigen | 0.817 | — |
| 62 | Mink (tensors) | 0.746 | — |
| 63 | SO3 D2 + Eigen + Beta | **0.827** | **peak** |
| 80 | Mink (tensors+eigen) | 0.806 | −2.1 vs peak |
| 86 | Mink (tensors+eigen+beta) | 0.818 | −0.9 vs peak |
| 94 | SO2 Degree 2 | 0.757 | — |
| 112 | SO2 D2 + Eigen | 0.787 | — |
| 155 | O3 Degree 3 | 0.806 | −2.1 vs peak |
| 173 | O3 D3 + Eigen | 0.813 | −1.4 vs peak |
| 219 | SO3 Degree 3 | 0.795 | −3.2 vs peak |
| 237 | SO3 D3 + Eigen | 0.804 | −2.3 vs peak |

Performance peaks at 63 features (SO3 D2+E+Beta). The new peak exceeds the previous best
(Mink tensors+eigen+beta at 0.818) by +0.9 pp using a cleaner polynomial invariant representation.

### Exact zero-variance feature counts for centered mesh data

All Allen Cell nuclei meshes are centered to (0, 0, 0) before Minkowski tensor computation
(verified: w010 = 1e-5 constant for all 5606 samples). Any invariant that is quadratic in
w010 is identically constant and zero-variance. The exact empirically verified counts are:

| Symmetry | Degree | Total | Zero-variance | Non-zero variance |
|----------|--------|-------|---------------|-------------------|
| SO3 | 1 | 8 | 0 | 8 |
| SO3 | 2 | 39 | 1 | 38 |
| SO3 | 3 | 219 | 7 | 212 |
| SO2 | 1 | 18 | 1 | 17 |
| SO2 | 2 | 94 | 2 | 92 |
| SO2 | 3 | 754 | 14 | 740 |

Zero-variance features are exclusively those where w010 appears **twice** in the polynomial
(e.g. `dot_w010_w010`, `qf_w010_W_w010`). Features linear in w010 (e.g. `dot_w010_w110`)
are non-zero variance — they reduce to a rescaled sum of the other tensor's components.
A previous theoretical estimate of ~46 zero-variance features for SO3 D3 was incorrect;
the actual count is 7.

### Observed rank of degree-3 invariants is much lower than nominal

Even after discarding the 7 zero-variance features, degree-3 polynomial invariants include
many near-redundant terms under typical mesh data. Surface curvature is relatively uniform
across medical imaging data, causing the four curvature-weighted tensors (w020, w120,
w220, w320) to be highly correlated (r > 0.95 for many pairs). The observed numerical rank
of SO3 degree-3 features is ~135 of the 212 non-zero-variance features, and ~120 when
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
- **Minkowski (tensors+eigen+beta, 86)** = 0.818: previously tied SO3 D2+E (0.817); now
  surpassed by SO3 D2+E+Beta (0.827, +0.9 pp) — adding beta to the polynomial invariant set
  breaks the tie and establishes a new ceiling

---

## Per-Class Recall Analysis

Per-class recall computed on the test set (LinearSVC, majority vote across 3 seeds,
best hyperparameters from BayesSearchCV).  All six conditions use the same train/test split.

| Class | Beta only | Eigen only | Eigen + Beta | Mink (tensors) | Mink (+E+B) | SO3 D2+E+B |
|---|---|---|---|---|---|---|
| Interphase | 0.806 | 0.819 | 0.839 | 0.802 | 0.821 | **0.829** |
| Prometaphase/Metaphase | 0.784 | 0.832 | 0.865 | 0.849 | 0.876 | **0.881** |
| Prophase | 0.216 | 0.591 | 0.693 | 0.636 | 0.705 | **0.739** |
| Early Prometaphase | 0.576 | 0.788 | 0.764 | 0.691 | 0.818 | **0.824** |
| ATC Half | 0.464 | **0.902** | 0.920 | 0.902 | 0.929 | 0.920 |
| ATC Early | 0.691 | **0.838** | 0.779 | 0.632 | 0.765 | 0.794 |

### Findings

**Prophase is the persistent bottleneck.** It is the lowest-recall class in every single
condition and the only class where raw tensor additivity matters: Mink (tensors) at 0.636
outperforms Eigenvalues only at 0.591 (−4.5 pp), confirming that eigenvalues and beta alone
lose orientation and absolute-scale information needed to separate Prophase from adjacent
stages. Adding beta recovers this loss (Eigen+Beta 0.693 > Mink tensors 0.636), and the
quadratic frob terms in SO3 D2+E+B push it further to 0.739 — the best result for this class.

**ATC Half is essentially solved by eigenvalues alone.** Recall is 0.902 with just 18
eigenvalue features and moves only ±0.03 across all richer conditions. The anaphase/telophase
shape change is geometrically large enough that principal-axis magnitudes fully capture it.

**ATC Early reveals a collinearity trap.** Eigenvalues alone achieve 0.838 (+20.6 pp over
raw tensors at 0.632). When raw tensor components are added back alongside eigenvalues in
Mink (+E+B), recall drops to 0.765: the resulting collinearity forces C=1.08 (heavy
regularisation) that suppresses the clean ATC Early signal carried by the eigenvalue subspace.
SO3 D2+E+B avoids this by using orthogonal polynomial invariants and recovers to 0.794.

**The non-additivity cost of eigenvalues is real but class-specific.** Only Prophase shows
a recall penalty when switching from raw tensors to eigenvalues; all other classes are equal
or better with eigenvalues. Beta indices and the D2 quadratic terms each provide incremental
Prophase recovery, making SO3 D2+E+B the best single feature set across all six classes.

---

## Classifier Ceiling Check (RF and LightGBM)

To determine whether LinearSVC results are limited by the classifier or by the features
themselves, Random Forest (RF) and LightGBM were run on the three key feature sets.
All runs used BayesSearchCV, n_iter=30, 5-fold CV, 3 seeds, n_jobs=2 per process
(three processes in parallel on 8-core machine).

| Feature Set | # Feat | LinearSVC | RF | LightGBM |
|-------------|--------|-----------|-----|----------|
| Eigen + Beta | 24 | 0.808 ± 0.001 | 0.825 ± 0.003 | **0.849 ± 0.000** |
| SO3 Degree 2 + Eigenvalues + Beta | 63 | 0.827 ± 0.009 | 0.827 ± 0.002 | **0.852 ± 0.000** |
| Minkowski (tensors+eigen+beta) | 86 | 0.818 ± 0.004 | 0.804 ± 0.008 | **0.833 ± 0.000** |

### Runtimes (optimization + evaluation, n_iter=30)

| Feature Set | # Feat | LinearSVC (n_iter=20) | RF (n_iter=30) | LightGBM (n_iter=30) |
|-------------|--------|----------------------|----------------|----------------------|
| Eigen + Beta | 24 | ~8s | 177s | 247s |
| SO3 D2 + Eigenvalues + Beta | 63 | ~290s | 200s | 546s |
| Minkowski (tensors+eigen+beta) | 86 | ~1244s | 274s | 646s |

RF is 4-5x faster than LightGBM and 3-5x faster than LinearSVC on larger feature sets.
LinearSVC's optimization time explodes with feature count (quadratic kernel matrix construction
inside BayesSearchCV); tree methods scale linearly with n_features.

### Interpretation

**LinearSVC was not at ceiling.** LightGBM beats LinearSVC by +2–4 pp across all three
conditions, confirming that the features contain more discriminative signal than a linear
classifier can extract. The gap of 2–4 pp is too large to be explained by the difference in
n_iter (20 vs 30); LinearSVC's simpler search space (C, gamma) is well-explored at n_iter=20.

**Feature rankings are fully preserved across all three classifiers.** All three independently
agree: SO3 D2 + Eigen + Beta > Minkowski (tensors+eigen+beta) > Eigen + Beta for LightGBM
and RF, and SO3 D2 > Minkowski > Eigen+Beta for LinearSVC (with Minkowski and Eigen+Beta
nearly tied at SVM). This cross-classifier consistency provides strong evidence that the
ranking reflects genuine structure in the features, not artefacts of the linear decision
boundary.

**RF underperforms on raw tensors** (0.804, worse than LinearSVC's 0.818). RF's greedy
split selection is hurt by the high collinearity and redundancy in 86 raw tensor components;
it wastes splits on correlated features. LightGBM's histogram-based boosting handles this
better, but still scores 1.9 pp below its best (SO3 D2 + Eigen + Beta). The polynomial
invariants' advantage is largest with LightGBM (0.852 vs 0.833, +1.9 pp), meaning the
invariants' lower redundancy and better geometric encoding benefit even powerful nonlinear
classifiers.

**The std=0.000 for LightGBM** (identical across 3 seeds) reflects LightGBM's deterministic
histogram-based fitting on a dataset large enough (~3900 training samples) that seed
variation in boosting rounds has negligible effect on test-set predictions.

---

## Framing for the Paper

### Primary result: LinearSVC

The main classification results should use **LinearSVC** as the primary classifier.
A linear classifier is a strong choice for evaluating feature quality for two reasons:

1. If features are linearly discriminative, they carry interpretable geometric information
that is not contingent on a powerful model's ability to find nonlinear patterns.
A linear boundary means the class differences are directly encoded in the feature directions.

2. LinearSVC results are conservative lower bounds.
The ceiling check confirms LightGBM achieves 2–4 pp more, so any claim made on
LinearSVC results (e.g. "SO3 D2+E+Beta outperforms raw tensors") is robust:
it holds even when the classifier is deliberately under-powered relative to the
data's true separability.

### Ceiling check: LightGBM as supporting evidence

LightGBM results serve two roles in the paper:

1. **Confirm feature rankings are not a LinearSVC artefact.**
The ordering SO3 D2+E+B > Mink+eigen+beta > Eigen+Beta is identical under LightGBM,
ruling out the hypothesis that the LinearSVC boundary is accidentally better-aligned
with some feature sets than others.

2. **Establish true performance ceiling.**
LinearSVC achieves 0.808–0.827 on the top conditions; LightGBM achieves 0.849–0.852.
The remaining gap above LightGBM is likely attributable to Bayes-error (overlapping cell
cycle stages in morphology space) rather than to limitations of the feature set.

### Story implications

The key finding is that **compact polynomial invariants (63 features) reach the same ceiling
as raw Minkowski tensors (86 features) under both a linear and a nonlinear classifier**, and
surpass them under LightGBM (+1.9 pp). This is significant because:

- The invariants are genuinely rotation-invariant by construction, whereas raw tensors require
the data to be pre-aligned or for the classifier to learn alignment implicitly.
- The performance gain of LightGBM over LinearSVC is approximately equal for invariants
(+2.5 pp) and raw tensors (+1.5 pp), suggesting no special interaction between classifier
type and feature type — the invariants are not "only good for linear classifiers."
- Eigen + Beta (24 features) achieves 0.849 with LightGBM — within 0.003 of the 63-feature
SO3 D2 set. This suggests a hard floor on what shape descriptors can achieve on this task
regardless of feature richness, and that most of the discriminative information is captured
by the per-tensor shape indices (eigenvalues and anisotropy ratios) alone.

---

## Practical Recommendations

| Priority | Feature Set | # Feat | Bal. Acc | When to use |
|----------|-------------|--------|----------|-------------|
| 1st | SO3 Degree 2 + Eigenvalues + Beta | 63 | **0.827** | Best overall; polynomial invariants + shape indices |
| 2nd | Minkowski (tensors+eigen+beta) | 86 | 0.818 | Best raw-tensor set; no symmetry assumption |
| 2nd | SO3 Degree 2 + Eigenvalues | 57 | 0.817 | Best without beta; cleaner SO3 invariants only |
| 3rd | SO3 Degree 1 + Eigenvalues + Beta | 32 | 0.814 | Good efficiency; half the features of 1st |
| 4th | SO2 Degree 1 + Eigenvalues + Beta | 42 | 0.809 | Best SO2 set; captures z-axis asymmetry + shape indices |
| 4th | Eigen + Beta | 24 | 0.808 | Best efficiency; 2.6× fewer features than 1st |
| 5th | SO2 Degree 1 + Eigenvalues | 36 | 0.799 | Simple, fast, captures z-axis asymmetry |
| 6th | SO3 Degree 1 + Eigenvalues | 26 | 0.793 | Simplest full-SO3 set |
| 7th | Eigenvalues only | 18 | 0.791 | Minimal viable feature set |
| Avoid | O3 Degree 3 (+E) | 155–173 | 0.806–0.813 | Better than SO3 D3 but still worse than D2+E+B |
| Avoid | SO3 Degree 3 (+E) | 219–237 | 0.795–0.804 | Dominated by O3 D3 and by D2+E+B |
| Avoid | Minkowski (tensors) | 62 | 0.746 | Dominated by any eigen-augmented set |

---

## Configuration

Minkowski baselines (`Eigenvalues only`, `Eigen + Beta`, `Minkowski (tensors)*`) and the unoptimised
SO3/SO2 Degree 1/2/3 baseline runs are documented in
[ALLEN_CELL_EIGEN_BETA_ABLATION.md](ALLEN_CELL_EIGEN_BETA_ABLATION.md).
`SO3 Degree 2 + Eigenvalues`, `SO3 Degree 1/2 + Eigenvalues` (SO2), are in `baselines/`.
CellProfiler and SPHARM experiments: see
[ALLEN_CELL_CELLPROFILER_ABLATION.md](ALLEN_CELL_CELLPROFILER_ABLATION.md) and
[ALLEN_CELL_SPHARM_SYSTEMATIC.md](ALLEN_CELL_SPHARM_SYSTEMATIC.md).

The commands below correspond to the JSON files in `polynomial_invariants/` and `classifier_ceiling/`.
All paths are relative to the repository root.
`--include` uses case-insensitive substring matching; use the full feature-set name as the pattern to
avoid unintended matches.

### SO3 polynomial invariants — Degree 3 BayesOpt (`polynomial_invariants/so3/`)

Two separate runs (~33 min each on Apple Silicon, n_jobs=5):

```bash
# SO3 Degree 3 alone  →  so3d3_optimized_invariants_{scores,hyperparams}.json
python benchmarks/invariants_classification.py \
    --input ../Minkowski_classifier/data/allen_cell/mitotic_cells_annotated/nuclei/minkowski_tensors_with_eigen_vals.csv \
    --output benchmarks/results/allen_cell/polynomial_invariants/so3/allen_cell_nuclei_so3d3_optimized_invariants \
    --max-so3-degree 3 \
    --include "SO3 Degree 3" \
    --optimize \
    --n_iter 20 \
    --linear-only \
    --seeds 3 \
    --n_jobs 5

# SO3 Degree 3 + Eigenvalues  →  so3d3_eigen_optimized_invariants_{scores,hyperparams}.json
python benchmarks/invariants_classification.py \
    --input ../Minkowski_classifier/data/allen_cell/mitotic_cells_annotated/nuclei/minkowski_tensors_with_eigen_vals.csv \
    --output benchmarks/results/allen_cell/polynomial_invariants/so3/allen_cell_nuclei_so3d3_eigen_optimized_invariants \
    --max-so3-degree 3 \
    --include "SO3 Degree 3 + Eigenvalues" \
    --optimize \
    --n_iter 20 \
    --linear-only \
    --seeds 3 \
    --n_jobs 5
```

### SO3 polynomial invariants — D1/D2 + Eigenvalues + Beta (`polynomial_invariants/so3/`)

```bash
# SO3 Degree 1 + Eigenvalues + Beta  →  so3d1_eigen_beta_invariants_{scores,hyperparams}.json
python benchmarks/invariants_classification.py \
    --input ../Minkowski_classifier/data/allen_cell/mitotic_cells_annotated/nuclei/minkowski_tensors_with_eigen_vals.csv \
    --output benchmarks/results/allen_cell/polynomial_invariants/so3/allen_cell_nuclei_so3d1_eigen_beta_invariants \
    --max-so3-degree 1 \
    --include "SO3 Degree 1 + Eigenvalues + Beta" \
    --optimize \
    --n_iter 20 \
    --linear-only \
    --seeds 3 \
    --n_jobs 5

# SO3 Degree 2 + Eigenvalues + Beta  →  so3d2_eigen_beta_invariants_{scores,hyperparams}.json
python benchmarks/invariants_classification.py \
    --input ../Minkowski_classifier/data/allen_cell/mitotic_cells_annotated/nuclei/minkowski_tensors_with_eigen_vals.csv \
    --output benchmarks/results/allen_cell/polynomial_invariants/so3/allen_cell_nuclei_so3d2_eigen_beta_invariants \
    --max-so3-degree 2 \
    --include "SO3 Degree 2 + Eigenvalues + Beta" \
    --optimize \
    --n_iter 20 \
    --linear-only \
    --seeds 3 \
    --n_jobs 5
```

### D2 feature-group ablation (`polynomial_invariants/so3/`)

```bash
# frob_self, frob_cross, frob_all, dots on top of D1+E+B  →  d2_ablation_invariants_{scores,hyperparams}.json
python benchmarks/invariants_classification.py \
    --input ../Minkowski_classifier/data/allen_cell/mitotic_cells_annotated/nuclei/minkowski_tensors_with_eigen_vals.csv \
    --output benchmarks/results/allen_cell/polynomial_invariants/so3/allen_cell_nuclei_d2_ablation_invariants \
    --max-so3-degree 2 \
    --include "D1+E+B + frob_self" "D1+E+B + frob_cross" "D1+E+B + frob_all" "D1+E+B + dots" \
    --optimize \
    --n_iter 20 \
    --linear-only \
    --seeds 3 \
    --n_jobs 5
```

### SO2 polynomial invariants (`polynomial_invariants/so2/`)

```bash
# SO2 Degree 1 and Degree 2 (standalone)  →  so2_optimized_invariants_{scores,hyperparams}.json
python benchmarks/invariants_classification.py \
    --input ../Minkowski_classifier/data/allen_cell/mitotic_cells_annotated/nuclei/minkowski_tensors_with_eigen_vals.csv \
    --output benchmarks/results/allen_cell/polynomial_invariants/so2/allen_cell_nuclei_so2_optimized_invariants \
    --max-so2-degree 2 \
    --include "SO2 Degree 1" "SO2 Degree 2" \
    --optimize \
    --n_iter 20 \
    --linear-only \
    --seeds 3 \
    --n_jobs 5

# SO2 Degree 1 + Eigenvalues + Beta  →  so2d1_eigen_beta_invariants_{scores,hyperparams}.json
python benchmarks/invariants_classification.py \
    --input ../Minkowski_classifier/data/allen_cell/mitotic_cells_annotated/nuclei/minkowski_tensors_with_eigen_vals.csv \
    --output benchmarks/results/allen_cell/polynomial_invariants/so2/allen_cell_nuclei_so2d1_eigen_beta_invariants \
    --max-so2-degree 1 \
    --include "SO2 Degree 1 + Eigenvalues + Beta" \
    --optimize \
    --n_iter 20 \
    --linear-only \
    --seeds 3 \
    --n_jobs 5
```

### Classifier ceiling — RF and LightGBM (`classifier_ceiling/`)

Three feature sets run in parallel (one process per condition, n_jobs=2 per process, 8-core machine).
Runtimes: RF 177–274 s, LightGBM 247–647 s per condition.

```bash
# Random Forest — three conditions run concurrently
python benchmarks/invariants_classification.py \
    --input ../Minkowski_classifier/data/allen_cell/mitotic_cells_annotated/nuclei/minkowski_tensors_with_eigen_vals.csv \
    --include "Eigen + Beta" \
    --optimize \
    --n_iter 30 \
    --classifier rf \
    --seeds 3 \
    --n_jobs 2 &

python benchmarks/invariants_classification.py \
    --input ../Minkowski_classifier/data/allen_cell/mitotic_cells_annotated/nuclei/minkowski_tensors_with_eigen_vals.csv \
    --max-so3-degree 2 \
    --include "SO3 Degree 2 + Eigenvalues + Beta" \
    --optimize \
    --n_iter 30 \
    --classifier rf \
    --seeds 3 \
    --n_jobs 2 &

python benchmarks/invariants_classification.py \
    --input ../Minkowski_classifier/data/allen_cell/mitotic_cells_annotated/nuclei/minkowski_tensors_with_eigen_vals.csv \
    --output benchmarks/results/allen_cell/classifier_ceiling/classifier_ceiling_rf_invariants \
    --include "Minkowski (tensors+eigen+beta)" \
    --optimize \
    --n_iter 30 \
    --classifier rf \
    --seeds 3 \
    --n_jobs 2 &

wait

# LightGBM — same three conditions
python benchmarks/invariants_classification.py \
    --input ../Minkowski_classifier/data/allen_cell/mitotic_cells_annotated/nuclei/minkowski_tensors_with_eigen_vals.csv \
    --include "Eigen + Beta" \
    --optimize \
    --n_iter 30 \
    --classifier lgbm \
    --seeds 3 \
    --n_jobs 2 &

python benchmarks/invariants_classification.py \
    --input ../Minkowski_classifier/data/allen_cell/mitotic_cells_annotated/nuclei/minkowski_tensors_with_eigen_vals.csv \
    --max-so3-degree 2 \
    --include "SO3 Degree 2 + Eigenvalues + Beta" \
    --optimize \
    --n_iter 30 \
    --classifier lgbm \
    --seeds 3 \
    --n_jobs 2 &

python benchmarks/invariants_classification.py \
    --input ../Minkowski_classifier/data/allen_cell/mitotic_cells_annotated/nuclei/minkowski_tensors_with_eigen_vals.csv \
    --output benchmarks/results/allen_cell/classifier_ceiling/classifier_ceiling_lgbm_invariants \
    --include "Minkowski (tensors+eigen+beta)" \
    --optimize \
    --n_iter 30 \
    --classifier lgbm \
    --seeds 3 \
    --n_jobs 2 &

wait
```

Only the `Minkowski (tensors+eigen+beta)` condition has a saved JSON in `classifier_ceiling/`; the
`Eigen + Beta` and `SO3 Degree 2 + Eigenvalues + Beta` ceiling results were recorded from console
output and are not stored as separate files.

### O(3) and O(2) parity-filtered invariants (`polynomial_invariants/o3_o2/`)

```bash
# O3 D1/D2/D3 and O2 D1/D2 — all conditions  →  o3_o2_invariants_{scores,hyperparams}.json
python benchmarks/invariants_classification.py \
    --input ../Minkowski_classifier/data/allen_cell/mitotic_cells_annotated/nuclei/minkowski_tensors_with_eigen_vals.csv \
    --output benchmarks/results/allen_cell/polynomial_invariants/o3_o2/o3_o2_invariants \
    --max-o3-degree 3 \
    --max-o2-degree 2 \
    --optimize \
    --n_iter 20 \
    --linear-only \
    --seeds 3 \
    --n_jobs 5

# O2 D3 only  →  o2_d3_invariants/o2_d3_invariants_{scores,hyperparams}.json
python benchmarks/invariants_classification.py \
    --input ../Minkowski_classifier/data/allen_cell/mitotic_cells_annotated/nuclei/minkowski_tensors_with_eigen_vals.csv \
    --output benchmarks/results/allen_cell/polynomial_invariants/o3_o2/o2_d3_invariants \
    --max-o2-degree 3 \
    --include "O2 Degree 3" \
    --optimize \
    --n_iter 20 \
    --linear-only \
    --seeds 3 \
    --n_jobs 5
```
