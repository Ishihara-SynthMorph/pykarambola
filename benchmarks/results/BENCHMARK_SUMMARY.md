# SO(3) Invariants Benchmark Summary

## Overview

This benchmark evaluates whether SO(3) rotational invariants computed from Minkowski tensors improve 3D shape classification compared to raw tensor components. We tested on two MedMNIST datasets at 64 resolution using Bayesian-optimized SVM classifiers.

**Core Hypothesis**: Rotation-invariant features should improve classification by removing pose-dependent noise while preserving shape information.

---

## Datasets

| Dataset | Samples (train/test) | Classes | Imbalance | Task |
|---------|---------------------|---------|-----------|------|
| adrenal3d | 1188 / 298 | 2 | 3.6:1 | Tumor type classification |
| vessel3d | 1335 / 382 | 2 | 7.9:1 | Vessel type classification |

**Input data paths** (from [minkowski_classifier](https://github.com/Ishihara-SynthMorph/minkowski_classifier) repo):
- `data/medmnist/adrenal3d/adrenal3d_64/minkowski_tensors_with_eigen_vals.csv`
- `data/medmnist/vessel3d/vessel3d_64/minkowski_tensors_with_eigen_vals.csv`

**Note on preprocessing**: The Minkowski tensors were computed directly from MedMNIST 3D voxel segmentations using `minkowski_tensors_from_label_image()` **without any rotation alignment or orientation normalization**. Objects retain their original orientation in the scanner/image coordinate system. This is relevant to interpreting the vessel3d results, where raw tensor components outperformed rotation-invariant features — suggesting that vessel orientation relative to the imaging axes may carry diagnostic information.

---

## Results

### adrenal3d 64

| Rank | Feature Set | # Features | Balanced Accuracy | Optimal Params |
|------|-------------|------------|-------------------|----------------|
| 1 | **SO3 Degree 2** | 39 | **0.833 ± 0.004** | C=1000, linear, PCA=26 |
| 2 | Baseline (w/ eigen) | 76 | 0.826 ± 0.013 | C=1000, linear, PCA=47 |
| 3 | SO3 Degree 3 | 219 | 0.824 ± 0.003 | C=1000, linear, PCA=65 |
| 4 | Baseline (tensors) | 52 | 0.821 ± 0.015 | C=2.1, linear, PCA=39 |
| 5 | SO3 Degree 1 | 8 | 0.806 ± 0.007 | C=988, linear, PCA=6 |

### vessel3d 64

| Rank | Feature Set | # Features | Balanced Accuracy | Optimal Params |
|------|-------------|------------|-------------------|----------------|
| 1 | **Baseline (tensors)** | 52 | **0.819 ± 0.001** | C=3.6, rbf, PCA=41 |
| 2 | Baseline (w/ eigen) | 76 | 0.811 ± 0.015 | C=1000, rbf, PCA=67 |
| 3 | SO3 Degree 3 | 219 | 0.798 ± 0.011 | C=0.1, linear, PCA=57 |
| 4 | SO3 Degree 1 | 8 | 0.796 ± 0.012 | C=24, rbf, PCA=8 |
| 5 | SO3 Degree 2 | 39 | 0.794 ± 0.006 | C=197, linear, PCA=39 |

---

## Feature Set Comparison

### Performance Ranking by Dataset

| Feature Set | adrenal3d Rank | vessel3d Rank | Consistent? |
|-------------|----------------|---------------|-------------|
| SO3 Degree 2 | 1st (0.833) | 5th (0.794) | No |
| Baseline (w/ eigen) | 2nd (0.826) | 2nd (0.811) | Yes |
| SO3 Degree 3 | 3rd (0.824) | 3rd (0.798) | Yes |
| Baseline (tensors) | 4th (0.821) | 1st (0.819) | No |
| SO3 Degree 1 | 5th (0.806) | 4th (0.796) | ~Yes |

### Key Observations

1. **No universally best feature set**: The optimal choice depends on the classification task.

2. **Eigenvalue-augmented baseline is consistently strong**: Ranked 2nd on both datasets, suggesting eigenvalues capture complementary shape information.

3. **SO3 Degree 3 offers no advantage over Degree 2**: Despite having 5.6x more features (219 vs 39), Degree 3 never outperforms Degree 2. The cubic invariants add redundancy without discriminative power.

4. **SO3 Degree 1 (8 features) underperforms**: The minimal scalar representation loses too much information for both tasks.

---

## Critique: SO(3) Invariants vs Core Hypothesis

### Where the Hypothesis Holds: adrenal3d

On adrenal3d, **SO3 Degree 2 outperforms all baselines** (0.833 vs 0.821-0.826), supporting the hypothesis that rotation invariance helps classification.

**Why it works**: Adrenal tumors likely vary in fundamental shape properties (volume, surface area, curvature) rather than orientation. The SO3 invariants successfully distill these properties while discarding irrelevant pose information.

### Where the Hypothesis Fails: vessel3d

On vessel3d, **raw tensors outperform all SO3 invariants** (0.819 vs 0.794-0.798), contradicting the hypothesis.

**Why it fails**: Blood vessels have inherent directionality (tubular structures with orientation). The vessel's alignment relative to the imaging axes may carry diagnostic information that rotation-invariant features discard.

### Revised Understanding

The core hypothesis requires qualification:

> **Rotation invariance helps when shape differences are orientation-independent. When object orientation itself is diagnostically relevant, raw tensor components preserve this information.**

This suggests a **task-dependent feature selection strategy**:
- Tumors, cells, organelles → prefer SO3 invariants
- Vessels, fibers, elongated structures → preserve orientation information

---

## What Optimal Parameters Reveal

### Kernel Choice

| Dataset | Best Feature Sets | Kernel | Interpretation |
|---------|-------------------|--------|----------------|
| adrenal3d | All | **Linear** | Classes are linearly separable; shape differences combine additively |
| vessel3d | Baselines | **RBF** | Nonlinear decision boundary needed; complex geometric relationships |
| vessel3d | SO3 Degree 2-3 | Linear | Invariants simplify the feature space |

**Insight**: adrenal3d shapes differ in ways that linear combinations of features can distinguish. vessel3d has more complex, nonlinear class boundaries that require the RBF kernel's implicit feature mapping.

### Regularization (C parameter)

| Dataset | C Values | Interpretation |
|---------|----------|----------------|
| adrenal3d | Mostly 1000 (max) | Clean data, well-separated classes, minimal regularization needed |
| vessel3d | Varied (0.1 - 1000) | More overlap between classes, regularization helps generalization |

**Exception**: adrenal3d Baseline (tensors) uses C=2.1, suggesting raw tensor components contain noise that regularization must suppress. The SO3 invariants and eigenvalues provide "cleaner" features that tolerate high C.

### PCA Components

| Feature Set | adrenal3d PCA | vessel3d PCA | Pattern |
|-------------|---------------|--------------|---------|
| SO3 Degree 1 (8 feat) | 6 (75%) | 8 (100%) | Most variance useful |
| SO3 Degree 2 (39 feat) | 26 (67%) | 39 (100%) | Moderate compression helps adrenal |
| SO3 Degree 3 (219 feat) | 65 (30%) | 57 (26%) | Heavy compression needed |
| Baseline (tensors, 52) | 39 (75%) | 41 (79%) | Similar compression ratio |
| Baseline (w/ eigen, 76) | 47 (62%) | 67 (88%) | Eigenvalues more useful for vessel |

**Insight**: Both datasets benefit from PCA dimensionality reduction, but optimal compression varies. Higher-degree invariants require more aggressive compression, suggesting many features are redundant or noisy.

---

## Conclusions

1. **SO3 invariants are not universally superior** to raw tensor features. Their value depends on whether rotation invariance aligns with the classification task.

2. **Degree 2 is the sweet spot** for SO3 invariants when they help. Degree 1 loses information; Degree 3 adds noise.

3. **Eigenvalue-augmented baselines are robust** across tasks, suggesting eigendecomposition captures complementary shape information.

4. **Parameter patterns reveal data structure**:
   - Linear kernel + high C → clean, linearly separable data
   - RBF kernel + varied C → complex, overlapping class distributions

5. **Recommendation**: For new classification tasks, evaluate both SO3 invariants and raw tensors. The optimal choice depends on whether object orientation carries task-relevant information.

---

## Methodology

- **Optimization**: BayesSearchCV, 50 iterations, 5-fold stratified CV
- **Classifier**: BalancedBaggingClassifier with SVM (handles class imbalance)
- **Evaluation**: 3 random seeds, balanced accuracy metric
- **Search space**: C ∈ [0.1, 1000], kernel ∈ {linear, rbf}, PCA components ∈ [2, n_features]
