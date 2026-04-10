# Allen Cell Nuclei: SO(2) Invariants — Preliminary Results (RBF-SVC, Unoptimized)

## Overview

Preliminary benchmark introducing SO(2) invariant feature sets (degrees 1–3) alongside the previously tested SO(3) and baseline feature sets.
Results use **default hyperparameters** (PCA=min(10, n_features), RBF-SVC C=1.0) — no Bayesian optimization.

**Classifier**: RBF-SVC (consistent with `ALLEN_CELL_PRELIMINARY.md`)
**Dataset**: Non-rotated nuclei only (`nuclei/`)
**Note**: Tr() columns included in baseline feature sets (62/86 features). The original preliminary used 52/76 features (Tr() excluded for cross-dataset comparison); SO(3) invariant scores are unaffected and match.

---

## Results

| Rank | Feature Set | # Features | Balanced Accuracy | Geo. Mean |
|------|-------------|------------|-------------------|-----------|
| 1 | Baseline (w/ eigen) | 86 | **0.725 ± 0.004** | **0.716 ± 0.006** |
| 2 | **SO2 Degree 1** | **18** | **0.716 ± 0.002** | **0.704 ± 0.002** |
| 3 | SO3 Degree 2 | 39 | 0.715 ± 0.003 | 0.705 ± 0.003 |
| 4 | SO3 Degree 1 | 8 | 0.703 ± 0.004 | 0.688 ± 0.004 |
| 5 | SO2 Degree 2 | 94 | 0.682 ± 0.002 | 0.670 ± 0.001 |
| 6 | Baseline (tensors) | 62 | 0.671 ± 0.005 | 0.645 ± 0.007 |
| 7 | SO3 Degree 3 | 219 | 0.647 ± 0.001 | 0.636 ± 0.000 |
| 8 | SO2 Degree 3 | 754 | 0.588 ± 0.013 | 0.564 ± 0.017 |

Total runtime: **38 seconds** (n_jobs=5, Apple Silicon)

---

## SO(2) vs SO(3) Comparison by Degree

| Degree | SO3 Bal. Acc | SO3 # Feat | SO2 Bal. Acc | SO2 # Feat | Δ (SO2 − SO3) |
|--------|-------------|------------|-------------|------------|----------------|
| 1 | 0.703 | 8 | **0.716** | 18 | **+0.013** |
| 2 | **0.715** | 39 | 0.682 | 94 | −0.033 |
| 3 | **0.647** | 219 | 0.588 | 754 | −0.060 |

---

## Interpretation

### SO2 Degree 1 is the standout result

With only 18 features, SO2 Degree 1 (0.716) ranks 2nd overall — essentially tied with SO3 Degree 2 (0.715, 39 features) and only 0.9 pp behind Baseline (w/ eigen) (0.725, 86 features).
This is the only case where SO(2) outperforms SO(3) at the same degree (+1.3 pp).

The SO(2) degree-1 feature set captures z-axis scalars: v_z components of rank-1 tensors, and both Tr(M)/3 and M_zz of rank-2 tensors.
For non-rotated nuclei imaged with a fixed microscope axis, M_zz and v_z encode real structural information about the nucleus relative to the imaging direction — information that SO(3) degree-1 scalars (traces only) discard.
This confirms that the z-axis carries discriminative signal in this dataset.

### SO2 Degree 2 drops below SO3 Degree 2

At degree 2, SO(2) adds inner products of |m|=1 and |m|=2 doublets (94 features total vs 39 for SO3).
The larger feature set is more aggressively compressed by PCA=10 (11% retained vs 26% for SO3 Degree 2), likely explaining the −3.3 pp gap.
This is a compression artefact; the signal content of SO2 Degree 2 may be competitive after optimization.

### Degree 3 hurts both symmetries (PCA=10 bottleneck)

SO3 Degree 3 (219 features) and SO2 Degree 3 (754 features) both rank in the bottom two.
The degree-3 results are not informative without optimization — PCA=10 discards 95–99% of features.
SO2 Degree 3 also shows higher variance (± 0.013) reflecting instability under extreme compression.

### Baseline scores consistent with previous preliminary

SO3 invariant scores match `ALLEN_CELL_PRELIMINARY.md` exactly (0.703, 0.715, 0.647).
Baseline (tensors) is slightly lower here (0.671 vs 0.685) due to 10 additional Tr() columns increasing compression at PCA=10; Baseline (w/ eigen) is very close (0.725 vs 0.732).

---

## Next Steps

Priority candidates for Bayesian-optimized follow-up:

1. **SO2 Degree 1** (18 features) — strongest new result; may improve further with optimized C and PCA
2. **SO2 Degree 2** (94 features) — likely penalised by PCA=10; optimization may close the gap with SO3 Degree 2
3. **SO3/SO2 Degree 3** — uninformative at default PCA; needs optimization to be fairly evaluated

---

## Configuration

```bash
python benchmarks/invariants_classification.py \
    --input ../Minkowski_classifier/data/allen_cell/mitotic_cells_annotated/nuclei/minkowski_tensors_with_eigen_vals.csv \
    --output benchmarks/results/allen_cell_nuclei_so2_preliminary_rbf \
    --max-so3-degree 3 \
    --max-so2-degree 3 \
    --seeds 3 \
    --n_jobs 5
```
