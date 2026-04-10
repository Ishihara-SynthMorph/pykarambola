# Allen Cell Nuclei: SO(2) Invariants — Preliminary Analysis Plan

## Objective

Extend the Allen Cell nuclei benchmark to include SO(2) polynomial invariants (degrees 1–3), which have not been tested on this dataset.
The analysis provides a first comparison of SO(2) vs SO(3) invariant feature sets alongside the raw tensor baselines.

---

## Dataset

| Item | Value |
|------|-------|
| Dataset | Allen Cell mitotic nuclei (non-rotated) |
| Minkowski tensors | `Minkowski_classifier/data/allen_cell/mitotic_cells_annotated/nuclei/minkowski_tensors_with_eigen_vals.csv` |
| Rotation | None (`nuclei/`; not `nuclei_rotated_only_in_xy/`) |

---

## Feature Sets

| # | Name | Symmetry | Degree | # Features | Status |
|---|------|----------|--------|------------|--------|
| 1 | Baseline (tensors) | — | — | 62 | Previously tested |
| 2 | Baseline (w/ eigen) | — | — | 86 | Previously tested |
| 3 | SO3 Degree 1 | SO(3) | 1 | 8 | Previously tested |
| 4 | SO3 Degree 2 | SO(3) | 2 | 39 | Previously tested |
| 5 | SO3 Degree 3 | SO(3) | 3 | 219 | Previously tested (unoptimized only) |
| 6 | SO2 Degree 1 | SO(2) | 1 | 18 | **New** |
| 7 | SO2 Degree 2 | SO(2) | 2 | 94 | **New** |
| 8 | SO2 Degree 3 | SO(2) | 3 | 754 | **New** |

Feature counts are based on the 14 tensors present in the dataset:
`w000`, `w010`, `w020`, `w100`, `w102`, `w110`, `w120`, `w200`, `w202`, `w210`, `w220`, `w300`, `w310`, `w320`.

### SO(2) invariant structure

SO(2) treats rotation about the z-axis as the symmetry group, decomposing tensors by charge m:

| Component | m | Source | Degree-1 output |
|-----------|---|--------|-----------------|
| Rank-0 scalars | 0 | `w000`, `w100`, `w200`, `w300` | 4 scalars |
| Rank-1 v_z | 0 | `w010_z`, `w110_z`, `w210_z`, `w310_z` | 4 scalars |
| Rank-2 Tr(M)/3 and M_zz | 0 | `w020`, `w120`, `w220`, `w320`, `w102`, `w202` | up to 10 scalars (dedup applied) |
| Rank-1 [v_x, v_y] | \|m\|=1 | `w010_xy`, … | doublets |
| Rank-2 [M_xz, M_yz] | \|m\|=1 | `w020_xz`, … | doublets |
| Rank-2 [M_xx−M_yy, 2M_xy] | \|m\|=2 | `w020_m2`, … | doublets |

Degree-2 adds inner products of same-charge doublets; degree-3 adds triple products coupling two |m|=1 doublets with one |m|=2 doublet.

---

## Classifier and Evaluation Protocol

This is a **preliminary** (no Bayesian optimization) run to establish baseline numbers quickly.

| Setting | Value |
|---------|-------|
| Classifier | LinearSVC (liblinear) |
| PCA components | min(10, n_features) — default |
| C | 1.0 — default |
| Seeds | 3 |
| Metrics | Balanced accuracy, geometric mean |
| Optimization | None (default hyperparameters) |

Rationale: default settings suffice to compare relative ordering of feature sets before committing to a full Bayesian optimization run, which would take significantly longer for SO2 Degree 3 (754 features).

---

## Run Command

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

Note: `--optimize` is intentionally omitted for this preliminary run.

---

## Key Questions

1. Do SO(2) invariants outperform SO(3) invariants on non-rotated data?
   SO(2) encodes z-axis orientation, which may be informative for nuclei imaged with a fixed microscope axis.

2. How do SO2 degrees compare to SO3 counterparts?
   SO3 Degree 3 (219 features) scored 0.647 unoptimized — well below Degree 2 (0.715), likely due to PCA=10 discarding most signal.
   With 754 features, SO2 Degree 3 faces an even steeper compression problem under default settings.

3. How does SO2 Degree 2 (94 features) compare to SO3 Degree 2 (39 features)?
   Both are degree-2 polynomial invariants but under different symmetry groups.

4. Is SO2 Degree 3 (754 features) usable with default PCA=10, or does it need a full optimization run?

---

## Next Step

After reviewing preliminary results, run a targeted Bayesian-optimized evaluation for the feature sets that show competitive performance.
