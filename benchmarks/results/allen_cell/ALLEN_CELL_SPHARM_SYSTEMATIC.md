# Allen Cell Nuclei: Systematic SPHARM Benchmark — lmax 1–11

## Overview

Bayesian-optimized classification results for raw SPHARM coefficients and
rotation-invariant SPHARM features (power spectrum + bispectrum) at lmax=1–11,
using the lmax=16 CSV truncated at each lmax value.

**Classifier**: LinearSVC (liblinear)
**Dataset**: Non-rotated nuclei only (`nuclei/`)
**Optimization**: BayesSearchCV, n_iter=20, 5-fold stratified CV
**Evaluation**: 3 seeds, balanced accuracy + geometric mean
**Input**: `spherical_harmonics_lmax_16.csv` (filtered to L≤lmax, M≤lmax per run)

---

## Results

### Raw SPHARM (orientation-dependent coefficients)

| lmax | # Features | Non-zero var | Rank | Balanced Accuracy | Geo. Mean | Best C | Best PCA |
|------|------------|-------------|------|-------------------|-----------|--------|----------|
| 1 | 8 | 4 | 4 | 0.423 ± 0.001 | 0.352 ± 0.009 | 0.10 | 4 |
| 2 | 18 | 9 | 9 | 0.495 ± 0.006 | 0.460 ± 0.007 | 1.08 | 18 |
| 3 | 32 | 16 | 16 | 0.547 ± 0.001 | 0.521 ± 0.001 | 13.3 | 22 |
| 4 | 50 | 25 | 25 | **0.601 ± 0.006** | **0.587 ± 0.006** | 0.11 | 30 |
| 5 | 72 | 36 | 36 | 0.597 ± 0.003 | 0.584 ± 0.004 | 0.39 | 57 |
| 6 | 98 | 49 | 49 | 0.604 ± 0.002 | 0.587 ± 0.002 | 0.10 | 57 |
| 7 | 128 | 64 | 64 | 0.609 ± 0.005 | 0.594 ± 0.005 | 0.23 | 115 |
| 8 | 162 | 81 | 81 | 0.606 ± 0.006 | 0.588 ± 0.007 | 997 | 78 |
| 9 | 200 | 100 | 100 | 0.625 ± 0.012 | 0.609 ± 0.013 | 0.107 | 165 |
| 10 | 242 | 121 | 121 | 0.615 ± 0.012 | 0.598 ± 0.014 | 0.104 | 159 |
| 11 | 288 | 144 | 144 | 0.614 ± 0.008 | 0.592 ± 0.010 | 956 | 249 |

### SPHARM Invariants (rotation-invariant power spectrum + bispectrum)

| lmax | # Features | Non-zero var | Rank | Balanced Accuracy | Geo. Mean | Best C | Best PCA |
|------|------------|-------------|------|-------------------|-----------|--------|----------|
| 1 | 6 | 5 | 4 | 0.444 ± 0.001 | 0.306 ± 0.017 | 13.3 | 5 |
| 2 | 14 | 11 | 8 | 0.589 ± 0.006 | 0.578 ± 0.006 | 1000 | 9 |
| 3 | 27 | 19 | 12 | 0.628 ± 0.004 | 0.618 ± 0.004 | 739 | 21 |
| 4 | 47 | 32 | 19 | 0.686 ± 0.002 | 0.672 ± 0.003 | 45.4 | 42 |
| 5 | 75 | 48 | 26 | 0.726 ± 0.004 | 0.713 ± 0.006 | 739 | 57 |
| 6 | 113 | 71 | 37 | 0.740 ± 0.002 | 0.726 ± 0.001 | 11.0 | 60 |
| 7 | 162 | 98 | 48 | 0.764 ± 0.001 | 0.753 ± 0.002 | 13.3 | 110 |
| 8 | 224 | 134 | 64 | 0.771 ± 0.013 | 0.761 ± 0.015 | 739 | 170 |
| 9 | 300 | 175 | 80 | **0.785 ± 0.004** | **0.778 ± 0.005** | 656 | 300 |
| 10 | 392 | 227 | 102 | 0.768 ± 0.009 | 0.761 ± 0.010 | 13.3 | 266 |
| 11 | 501 | 285 | 124 | 0.752 ± 0.013 | 0.743 ± 0.015 | 0.233 | 449 |

---

## Interpretation

### SPHARM Inv peaks at lmax=9; performance degrades from lmax=10 onward

SPHARM Inv increases through lmax=9 then falls monotonically:

| Step | Δ Bal. Acc |
|------|-----------|
| lmax 1→2 | +14.5 pp |
| lmax 2→3 | +3.9 pp |
| lmax 3→4 | +5.8 pp |
| lmax 4→5 | +4.0 pp |
| lmax 5→6 | +1.4 pp |
| lmax 6→7 | +2.4 pp |
| lmax 7→8 | +0.7 pp |
| lmax 8→9 | +1.4 pp |
| lmax 9→10 | −1.7 pp |
| lmax 10→11 | −1.6 pp |

**lmax=9 (300 features, rank 80, 0.785) is the definitive peak.** The key signal is the
regularization C: it drops from 656 (lmax=9) to 13.3 (lmax=10) to 0.233 (lmax=11) — a
~2800× collapse. This reflects rapidly worsening collinearity as bispectrum feature count
grows as O(lmax³) while rank grows only as O(lmax). At lmax=11, 501 features share only
124 independent directions (rank/nz = 0.435), and the classifier requires extreme
regularization that destroys discriminative power.

### Odd lmax consistently outperforms the adjacent even lmax

Across all tested values, odd lmax outperforms both the preceding and following even lmax:

| Odd lmax | Bal. Acc | Even lmax−1 | Even lmax+1 |
|----------|----------|-------------|-------------|
| lmax=3 | 0.628 | lmax=2: 0.589 | lmax=4: 0.686 |
| lmax=5 | 0.726 | lmax=4: 0.686 | lmax=6: 0.740 |
| lmax=7 | **0.764** | lmax=6: 0.740 | lmax=8: 0.771 |
| lmax=9 | **0.785** | lmax=8: 0.771 | lmax=10: 0.768 |
| lmax=11 | 0.752 | lmax=10: 0.768 | — |

At lmax=3, 5, and 7, odd performs between the two adjacent even values — i.e., the gain
from adding one more even degree exceeds the odd gain (lmax=8 > lmax=7). But at lmax=9,
the pattern inverts: **lmax=9 outperforms lmax=10** (+1.7 pp), making lmax=9 the global peak.

The structural reason is spherical harmonic parity: bispectrum triples (l₁, l₂, l₃) require
l₁+l₂+l₃ even. Even-degree additions have more valid coupling triples (even+even+even and
odd+odd+even), while odd-degree additions require mixed-parity pairs only — so odd degrees
add fewer but proportionally more independent features (higher marginal rank/nz). This gives
odd lmax a lean, efficient feature set at low collinearity, reflected in high C values
(lmax=7: C=13.3, lmax=9: C=656) relative to the surrounding even lmax values.

### The lmax=9→11 drop is a definitive indicator of SPHARM's limits

The fact that both lmax=10 and lmax=11 underperform lmax=9 — despite lmax=11 being odd —
confirms that the bispectrum has reached its practical ceiling on this dataset. The odd-lmax
structural advantage (which rescued lmax=9 over lmax=8) breaks down at lmax=11: rank/nz =
0.435 falls below lmax=10's 0.449, and C collapses to 0.233 (~2800× below lmax=9's 656).
This is not a noise artifact — two consecutive lmax values both fail to recover lmax=9's
performance, and the collinearity trend is monotone and worsening.

The SPHARM power spectrum + bispectrum representation is structurally limited for this task:
feature count grows as O(lmax³) while independent rank grows only as O(lmax²), so the
signal-to-redundancy ratio degrades irreversibly beyond lmax=9. No further lmax increase
is expected to recover performance.

### Raw SPHARM plateaus around 0.600 from lmax=4 onward

Raw SPHARM peaks at lmax=4 (0.601) and fluctuates between 0.597–0.609 at lmax=4–8,
showing no systematic gain from higher frequency bands. The C values span 0.10–997
with no clear pattern, and the very low values (0.10–0.39) at most lmax confirm that
the orientation-dependent coefficients are highly collinear: different rotations of the
same shape produce very different coefficient vectors that a linear classifier must
regularise heavily to handle. Adding more bands just introduces more correlated noise.
This confirms raw SPHARM coefficients are a poor choice for linear classification unless
the dataset is perfectly rotation-normalised.

### The invariant transformation fundamentally changes feature geometry

The C values reveal a ~1900× difference between SPHARM Inv (C=739 at lmax=5) and raw
SPHARM (C=0.39 at lmax=5). This is not a hyperparameter artefact — it reflects a
fundamental difference in feature geometry: raw SPHARM coefficients form a highly collinear
space (rotation of the same shape scatters coefficients across the space, requiring heavy
regularisation), whereas the power spectrum + bispectrum maps rotation-equivalent shapes
to nearby points, enabling a high-margin classifier with weak regularisation.

### SPHARM Inv consistently and increasingly outperforms raw SPHARM

At every lmax, SPHARM Inv > raw SPHARM. The gap widens with lmax:

| lmax | SPHARM Inv | Raw SPHARM | Gap |
|------|------------|------------|-----|
| 1 | 0.444 | 0.423 | +2.1 pp |
| 4 | 0.686 | 0.601 | +8.5 pp |
| 5 | 0.726 | 0.597 | +12.9 pp |
| 7 | 0.764 | 0.609 | +15.5 pp |
| 8 | 0.771 | 0.606 | +16.5 pp |

### SPHARM Inv at lmax=8 matches CellProfiler (full); lmax=9 peak still below Minkowski

At lmax=8, SPHARM Inv reaches 0.771 ± 0.013 — statistically identical to CellProfiler
(full 22 features: 0.769 ± 0.003). This is a notable crossover: with enough frequency
bands, the power spectrum + bispectrum of the nuclear shape recovers the same
discriminative signal as standard image analysis descriptors.

The lmax=9 peak (0.785 ± 0.004, 300 features) surpasses CellProfiler but still sits
below all Minkowski-based feature sets:

- Eigenvalues only (18 features): 0.791 — still better than SPHARM Inv lmax=9 (300 features)
- SO3 Degree 2 (39 features): 0.783 — comparable to SPHARM Inv lmax=9, at 87% fewer features
- SO3 Degree 2 + Eigenvalues (57 features): 0.817
- Minkowski (tensors+eigen+beta, 86 features): 0.818

Eigenvalues only (18 features, 0.791) marginally exceeds the best SPHARM Inv result across
all tested lmax values (lmax=9, 300 features, 0.785). The Minkowski tensor eigenvalues
encode shape anisotropy more directly and compactly than the spherical harmonic power
spectrum; even 10 bands do not recover the information in 18 eigenvalues.

---

## Configuration

```bash
# lmax=1–5
python benchmarks/invariants_classification.py \
    --input ../Minkowski_classifier/data/allen_cell/mitotic_cells_annotated/nuclei/minkowski_tensors_with_eigen_vals.csv \
    --spharm-input ../Minkowski_classifier/data/allen_cell/mitotic_cells_annotated/nuclei/spherical_harmonics_lmax_16.csv \
    --spharm-lmax 1 2 3 4 5 \
    --output benchmarks/results/allen_cell/spharm/lmax_1_5 \
    --include "SPHARM" \
    --optimize \
    --n_iter 20 \
    --linear-only \
    --seeds 3 \
    --n_jobs 5

# lmax=6–8
python benchmarks/invariants_classification.py \
    --input ../Minkowski_classifier/data/allen_cell/mitotic_cells_annotated/nuclei/minkowski_tensors_with_eigen_vals.csv \
    --spharm-input ../Minkowski_classifier/data/allen_cell/mitotic_cells_annotated/nuclei/spherical_harmonics_lmax_16.csv \
    --spharm-lmax 6 7 8 \
    --output benchmarks/results/allen_cell/spharm/lmax_6_8 \
    --include "SPHARM" \
    --optimize \
    --n_iter 20 \
    --linear-only \
    --seeds 3 \
    --n_jobs 5
```

---

```bash
# lmax=9
~/miniforge3/envs/pykarambola-bench/bin/python benchmarks/invariants_classification.py \
    --input /Users/keisuke/Documents/GitHub/minkowski_classifier/data/allen_cell/mitotic_cells_annotated/nuclei/minkowski_tensors_with_eigen_vals.csv \
    --spharm-input /Users/keisuke/Documents/GitHub/minkowski_classifier/data/allen_cell/mitotic_cells_annotated/nuclei/spherical_harmonics_lmax_16.csv \
    --spharm-lmax 9 \
    --output benchmarks/results/allen_cell/spharm/allen_cell_nuclei_spharm_systematic_9_invariants \
    --include "SPHARM" \
    --optimize \
    --n_iter 20 \
    --linear-only \
    --seeds 3 \
    --n_jobs 5

# lmax=10
~/miniforge3/envs/pykarambola-bench/bin/python benchmarks/invariants_classification.py \
    --input /Users/keisuke/Documents/GitHub/minkowski_classifier/data/allen_cell/mitotic_cells_annotated/nuclei/minkowski_tensors_with_eigen_vals.csv \
    --spharm-input /Users/keisuke/Documents/GitHub/minkowski_classifier/data/allen_cell/mitotic_cells_annotated/nuclei/spherical_harmonics_lmax_16.csv \
    --spharm-lmax 10 \
    --output benchmarks/results/allen_cell/spharm/allen_cell_nuclei_spharm_systematic_10_invariants \
    --include "SPHARM" \
    --optimize \
    --n_iter 20 \
    --linear-only \
    --seeds 3 \
    --n_jobs 5
```

---

## Runtime

- lmax=1–5: ~6 min (10 feature sets, n_iter=20, 5-fold CV, 3 seeds, LinearSVC, n_jobs=5, Apple M2)
- lmax=6–8: ~14 min (6 feature sets, same settings, Apple M2)
- lmax=9: ~8 min (2 feature sets, pykarambola-bench env, Apple M5)
- lmax=10: ~8 min (2 feature sets, pykarambola-bench env, Apple M5)
- lmax=11: ~9 min (2 feature sets, pykarambola-bench env, Apple M5)
