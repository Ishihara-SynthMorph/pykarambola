# Allen Cell Nuclei: SO(2) Invariants — Preliminary Results (Unoptimized)

## Overview

Preliminary benchmark introducing SO(2) invariant feature sets (degrees 1–3) alongside the previously tested SO(3) baselines.
Results use **default hyperparameters** (PCA=min(10, n_features), LinearSVC C=1.0) — no Bayesian optimization.

**Classifier**: LinearSVC (liblinear)
**Dataset**: Non-rotated nuclei only (`nuclei/`)
**Note**: Tr() columns included in baseline feature sets (62/86 features), consistent with `ALLEN_CELL_OPTIMIZED.md`.

---

## Results

| Rank | Feature Set | # Features | Balanced Accuracy | Geo. Mean |
|------|-------------|------------|-------------------|-----------|
| 1 | **SO3 Degree 2** | 39 | **0.682 ± 0.004** | **0.669 ± 0.005** |
| 2 | SO2 Degree 2 | 94 | 0.671 ± 0.009 | 0.658 ± 0.010 |
| 3 | SO3 Degree 1 | 8 | 0.662 ± 0.008 | 0.630 ± 0.012 |
| 4 | Baseline (w/ eigen) | 86 | 0.660 ± 0.005 | 0.622 ± 0.007 |
| 5 | SO2 Degree 1 | 18 | 0.646 ± 0.005 | 0.611 ± 0.009 |
| 6 | SO3 Degree 3 | 219 | 0.626 ± 0.005 | 0.590 ± 0.007 |
| 7 | Baseline (tensors) | 62 | 0.615 ± 0.004 | 0.570 ± 0.008 |
| 8 | SO2 Degree 3 | 754 | 0.578 ± 0.001 | 0.526 ± 0.003 |

Total runtime: **20 seconds** (n_jobs=5, Apple Silicon)

---

## SO(2) vs SO(3) Comparison by Degree

| Degree | SO3 Bal. Acc | SO3 # Feat | SO2 Bal. Acc | SO2 # Feat | Δ (SO2 − SO3) |
|--------|-------------|------------|-------------|------------|----------------|
| 1 | 0.662 | 8 | 0.646 | 18 | −0.016 |
| 2 | 0.682 | 39 | 0.671 | 94 | −0.011 |
| 3 | 0.626 | 219 | 0.578 | 754 | −0.048 |

SO(3) outperforms SO(2) at every degree under default settings.
SO(2) uses more features at each degree (18 vs 8, 94 vs 39, 754 vs 219) but scores lower — PCA=10 discards proportionally more signal from larger feature sets.

---

## Interpretation

### SO3 Degree 2 remains the best invariant feature set

At 39 features with PCA=10, SO3 Degree 2 retains a higher fraction of its feature space than any other set (10/39 = 26% retained vs 10/94 = 11% for SO2 Degree 2). This compression advantage likely accounts for most of the gap.

### SO2 Degree 2 is competitive and warrants optimization

SO2 Degree 2 (0.671) trails SO3 Degree 2 (0.682) by only 1.1 pp while using 2.4× more features.
Under optimization, the larger feature budget of SO2 Degree 2 may become an advantage rather than a liability — as seen with SPHARM Inv lmax=5 in the optimized run, which gained +5.6 pp from optimization.

### High-degree sets (Degree 3) are hurt by PCA=10

Both SO3 Degree 3 (219 features) and SO2 Degree 3 (754 features) drop below their lower-degree counterparts.
Degree 3 discards 95–99% of features at PCA=10 — these results are not informative without optimization.

### Baseline drops vs previous preliminary

The previous preliminary (ALLEN_CELL_PRELIMINARY.md) used RBF-SVC, while this run uses LinearSVC.
Baseline (tensors) drops from 0.685 (RBF) to 0.615 (Linear), confirming that the RBF kernel is important for this feature set at default C.
All comparisons within this table are internally consistent.

---

## Next Steps

Candidates for Bayesian-optimized follow-up, in priority order:

1. **SO2 Degree 2** (94 features) — closest competitor to SO3 Degree 2; optimization likely to close the gap
2. **SO3 Degree 3** (219 features) — already tested optimized for SO3 Degree 2; Degree 3 result at optimized PCA is unknown
3. **SO2 Degree 1** (18 features) — lightweight; may benefit from higher C
4. **SO2 Degree 3** (754 features) — high feature count; optimization needed but expensive

---

## Configuration

```bash
python benchmarks/invariants_classification.py \
    --input ../Minkowski_classifier/data/allen_cell/mitotic_cells_annotated/nuclei/minkowski_tensors_with_eigen_vals.csv \
    --output benchmarks/results/allen_cell_nuclei_so2_preliminary \
    --max-so3-degree 3 \
    --max-so2-degree 3 \
    --linear-only \
    --seeds 3 \
    --n_jobs 5
```
