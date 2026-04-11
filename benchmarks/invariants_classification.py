#!/usr/bin/env python3
"""
Benchmark SO(3) invariants for Adrenal3D classification.

This script compares classification performance of SO(3) invariants at different
polynomial degrees against baseline raw Minkowski tensor features.

Usage:
    python benchmarks/invariants_classification.py \
        --input ../minkowski_classifier/data/medmnist/adrenal3d/adrenal3d_28/minkowski_tensors_with_eigen_vals.csv \
        --optimize --seeds 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

# Add parent to path for pykarambola imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pykarambola import compute_invariants

# Optional imports with fallbacks
try:
    from imblearn.ensemble import BalancedBaggingClassifier
    from imblearn.metrics import geometric_mean_score
    HAS_IMBALANCED = True
except ImportError:
    HAS_IMBALANCED = False
    print("Warning: imbalanced-learn not installed. Using BaggingClassifier fallback.")

try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False
    print("Warning: scikit-optimize not installed. Optimization disabled.")


# Column definitions for tensor reconstruction
SCALAR_COLS = ['w000', 'w100', 'w200', 'w300']
VECTOR_TENSORS = ['w010', 'w110', 'w210', 'w310']
MATRIX_TENSORS = ['w020', 'w120', 'w220', 'w320', 'w102', 'w202']


def reconstruct_tensors(row: pd.Series) -> dict[str, np.ndarray | float]:
    """Reconstruct tensor dict from a flattened CSV row."""
    tensors = {}

    # Scalars
    for name in SCALAR_COLS:
        if name in row:
            tensors[name] = float(row[name])

    # Vectors (3,)
    for name in VECTOR_TENSORS:
        cols = [f'{name}_{i}' for i in range(3)]
        if all(c in row for c in cols):
            tensors[name] = np.array([row[c] for c in cols])

    # Symmetric matrices (3, 3) - stored as upper triangle
    for name in MATRIX_TENSORS:
        cols = {
            (0, 0): f'{name}_00', (0, 1): f'{name}_01', (0, 2): f'{name}_02',
            (1, 1): f'{name}_11', (1, 2): f'{name}_12', (2, 2): f'{name}_22',
        }
        if all(c in row for c in cols.values()):
            M = np.zeros((3, 3))
            for (i, j), col in cols.items():
                M[i, j] = row[col]
                if i != j:
                    M[j, i] = row[col]  # Symmetric
            tensors[name] = M

    return tensors


def build_invariant_features(
    df: pd.DataFrame,
    max_degree: int,
    symmetry: str = 'SO3',
) -> tuple[np.ndarray, list[str]]:
    """Compute invariant features for all samples.

    Returns feature matrix X and list of feature names.
    """
    invariants_list = []
    feature_names = None

    for idx, row in df.iterrows():
        tensors = reconstruct_tensors(row)
        inv = compute_invariants(tensors, max_degree=max_degree, symmetry=symmetry)

        if feature_names is None:
            feature_names = sorted(inv.keys())

        invariants_list.append([inv[name] for name in feature_names])

    X = np.array(invariants_list)
    return X, feature_names


def build_so3d2_so2d1_extra_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """SO3 Degree 2 + SO2 Degree 1 z-axis scalars not already in SO3 Degree 2.

    The extra SO2 Degree 1 features are the v_z components of rank-1 tensors
    ({name}_z) and the M_zz components of rank-2 tensors ({name}_zz), which
    encode z-axis orientation information absent from SO(3) invariants.
    """
    results = []
    so3_keys = None
    so2_extra_keys = None

    for _, row in df.iterrows():
        tensors = reconstruct_tensors(row)
        so3_inv = compute_invariants(tensors, max_degree=2, symmetry='SO3')
        so2_d1 = compute_invariants(tensors, max_degree=1, symmetry='SO2')

        if so3_keys is None:
            so3_keys = sorted(so3_inv.keys())
            so2_extra_keys = sorted(k for k in so2_d1 if k not in so3_inv)

        results.append([so3_inv[k] for k in so3_keys] + [so2_d1[k] for k in so2_extra_keys])

    feature_names = so3_keys + so2_extra_keys
    return np.array(results), feature_names


def build_so3d2_eigen_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """SO3 Degree 2 invariants + eigenvalues of rank-2 tensors from CSV.

    Eigenvalues are rotation invariants and add explicit shape information
    without the collinearity introduced by also including raw tensor components.
    Beta (anisotropy index) columns are excluded as they are derived from
    eigenvalues and would add redundancy.
    """
    eval_cols = sorted(c for c in df.columns if 'EVal' in c)
    results = []
    so3_keys = None

    for _, row in df.iterrows():
        tensors = reconstruct_tensors(row)
        so3_inv = compute_invariants(tensors, max_degree=2, symmetry='SO3')

        if so3_keys is None:
            so3_keys = sorted(so3_inv.keys())

        results.append([so3_inv[k] for k in so3_keys] + [row[c] for c in eval_cols])

    feature_names = so3_keys + eval_cols
    return np.array(results), feature_names


def build_invariants_eigen_features(
    df: pd.DataFrame,
    symmetry: str,
    degree: int,
) -> tuple[np.ndarray, list[str]]:
    """Polynomial invariants (symmetry, degree) + eigenvalues of rank-2 tensors.

    Eigenvalues add det(W) per tensor — I₃ = λ₁λ₂λ₃, a degree-3 quantity
    algebraically independent of all degree-≤2 polynomial invariants.
    Beta (anisotropy index) columns are excluded to avoid redundancy.
    """
    eval_cols = sorted(c for c in df.columns if 'EVal' in c)
    results = []
    inv_keys = None

    for _, row in df.iterrows():
        tensors = reconstruct_tensors(row)
        inv = compute_invariants(tensors, max_degree=degree, symmetry=symmetry)

        if inv_keys is None:
            inv_keys = sorted(inv.keys())

        results.append([inv[k] for k in inv_keys] + [row[c] for c in eval_cols])

    feature_names = inv_keys + eval_cols
    return np.array(results), feature_names


def build_eigen_only_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Eigenvalues of rank-2 tensors only (18 features)."""
    cols = sorted(c for c in df.columns if 'EVal' in c)
    return df[cols].values, cols


CP_POSITION_COLS = {
    'Image_Count_ConvertImageToObjects',
    'Mean_FilterObjects_AreaShape_BoundingBoxMaximum_X',
    'Mean_FilterObjects_AreaShape_BoundingBoxMaximum_Y',
    'Mean_FilterObjects_AreaShape_BoundingBoxMaximum_Z',
    'Mean_FilterObjects_AreaShape_BoundingBoxMinimum_X',
    'Mean_FilterObjects_AreaShape_BoundingBoxMinimum_Y',
    'Mean_FilterObjects_AreaShape_BoundingBoxMinimum_Z',
    'Mean_FilterObjects_AreaShape_Center_X',
    'Mean_FilterObjects_AreaShape_Center_Y',
    'Mean_FilterObjects_AreaShape_Center_Z',
    'Mean_FilterObjects_Location_Center_X',
    'Mean_FilterObjects_Location_Center_Y',
    'Mean_FilterObjects_Location_Center_Z',
    'Mean_FilterObjects_AreaShape_BoundingBoxVolume',
}


def build_cellprofiler_features(
    df: pd.DataFrame,
    cp_df: pd.DataFrame,
    shape_only: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Extract CellProfiler features, joined to df by image_num.

    shape_only=True excludes position, bounding box, count and centroid columns,
    retaining only the 8 pure shape descriptors.
    """
    merged = df[['image_num']].merge(cp_df, on='image_num', how='left')
    meta = {'image_num', 'label', 'subfolder'}
    if shape_only:
        cols = [c for c in cp_df.columns if c not in meta and c not in CP_POSITION_COLS]
    else:
        cols = [c for c in cp_df.columns if c not in meta]
    X = merged[cols].values
    return X, cols


def build_spharm_invariant_features(
    df: pd.DataFrame,
    spharm_df: pd.DataFrame,
    lmax: int = 5,
) -> tuple[np.ndarray, list[str]]:
    """Compute rotation-invariant power spectrum + bispectrum from SPHARM coefficients."""
    from pykarambola.spharm_invariants import compute_spharm_invariants
    merged = df[['image_num']].merge(spharm_df, on='image_num', how='left')
    return compute_spharm_invariants(merged, lmax=lmax)


def build_spharm_features(
    df: pd.DataFrame,
    spharm_df: pd.DataFrame,
    lmax: int | None = None,
    exclude_watertight: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Extract spherical harmonics features, joined to df by image_num.

    lmax filters to coefficients with L <= lmax AND M <= lmax, matching the
    aics-shparam storage convention where each L stores M=0..lmax_stored.
    """
    import re
    merged = df[['image_num']].merge(spharm_df, on='image_num', how='left')
    cols = [c for c in spharm_df.columns if c.startswith('shcoeffs_')]
    if lmax is not None:
        cols = [c for c in cols
                if int(re.search(r'L(\d+)', c).group(1)) <= lmax
                and int(re.search(r'M(\d+)', c).group(1)) <= lmax]
    if not exclude_watertight:
        wt = [c for c in spharm_df.columns if c == 'watertight_components']
        cols = cols + wt
    X = merged[cols].values
    return X, cols


def build_baseline_features(df: pd.DataFrame, mode: str = 'tensors') -> tuple[np.ndarray, list[str]]:
    """Extract baseline raw tensor features.

    mode='tensors'            — raw components + traces, no betas, no eigenvalues (62 features)
    mode='tensors'            — raw components + traces, no betas, no eigenvalues (62 features)
    mode='tensors+beta'       — tensors + betas, no eigenvalues (68 features)
    mode='tensors+eigen'      — tensors + eigenvalues, no betas (80 features)
    mode='tensors+eigen+beta' — all columns including betas (86 features)
    mode='beta'               — beta columns only (6 features)
    mode='eigen'              — eigenvalue columns only (18 features)  [alias for build_eigen_only_features]
    mode='eigen+beta'         — eigenvalues + betas only (24 features)
    """
    all_feature_cols = df.columns[3:].tolist()

    if mode == 'tensors+eigen+beta':
        cols = all_feature_cols
    elif mode == 'tensors+eigen':
        cols = [c for c in all_feature_cols if 'beta' not in c]
    elif mode == 'tensors+beta':
        cols = [c for c in all_feature_cols if 'EVal' not in c]
    elif mode == 'eigen+beta':
        cols = [c for c in all_feature_cols if c.startswith('beta') or 'EVal' in c]
    elif mode == 'beta':
        cols = [c for c in all_feature_cols if c.startswith('beta')]
    else:  # 'tensors'
        cols = [c for c in all_feature_cols if 'beta' not in c and 'EVal' not in c]

    X = df[cols].values
    return X, cols


def create_pipeline(n_components: int, C: float, kernel: str = 'rbf', gamma: str = 'scale',
                    use_balanced: bool = True, random_state: int = 0, n_jobs: int = -1,
                    linear_only: bool = False):
    """Create classification pipeline."""
    if linear_only:
        estimator = LinearSVC(C=C, dual=False, max_iter=1000)
    else:
        estimator = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)

    if use_balanced and HAS_IMBALANCED:
        classifier = BalancedBaggingClassifier(
            estimator=estimator,
            sampling_strategy='auto',
            replacement=False,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    else:
        from sklearn.ensemble import BaggingClassifier
        classifier = BaggingClassifier(
            estimator=estimator,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    return Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_components)),
        ('classifier', classifier),
    ])


def optimize_hyperparams(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_iter: int = 50,
    cv: int = 5,
    random_state: int = 0,
    n_jobs: int = -1,
    verbose: int = 1,
    linear_only: bool = False,
) -> dict:
    """Run Bayesian optimization to find best hyperparameters."""
    if not HAS_SKOPT:
        # Return default params
        return {
            'pca__n_components': min(10, X_train.shape[1]),
            'classifier__estimator__C': 1.0,
            'classifier__estimator__kernel': 'rbf',
            'classifier__estimator__gamma': 'scale',
        }

    if linear_only:
        # LinearSVC: only C and PCA to tune — much faster than SVC(kernel='linear')
        search_space = {
            'pca__n_components': Integer(2, min(X_train.shape[1], X_train.shape[0] - 1)),
            'classifier__estimator__C': Real(1e-1, 1e3, prior='log-uniform'),
        }
        estimator = LinearSVC(dual=False, max_iter=1000)
    else:
        search_space = {
            'pca__n_components': Integer(2, min(X_train.shape[1], X_train.shape[0] - 1)),
            'classifier__estimator__C': Real(1e-1, 1e3, prior='log-uniform'),
            'classifier__estimator__kernel': Categorical(['linear', 'rbf']),
            'classifier__estimator__gamma': Categorical(['scale', 'auto']),
        }
        estimator = SVC()

    # n_jobs=1 for inner classifier: BayesSearchCV already parallelises CV folds
    # via its own n_jobs; nested parallelism causes contention and slowdown.
    if HAS_IMBALANCED:
        classifier = BalancedBaggingClassifier(
            estimator=estimator,
            sampling_strategy='auto',
            replacement=False,
            random_state=random_state,
            n_jobs=1,
        )
    else:
        from sklearn.ensemble import BaggingClassifier
        classifier = BaggingClassifier(estimator=estimator, random_state=random_state, n_jobs=1)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA()),
        ('classifier', classifier),
    ])

    scoring = {'balanced_accuracy': 'balanced_accuracy'}
    if HAS_IMBALANCED:
        scoring['geometric_mean'] = make_scorer(geometric_mean_score)

    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    opt = BayesSearchCV(
        pipeline,
        search_space,
        n_iter=n_iter,
        cv=cv_splitter,
        scoring=scoring,
        refit='balanced_accuracy',
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=verbose,
    )

    opt.fit(X_train, y_train)
    return opt.best_params_


def evaluate_single_run(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    params: dict,
    random_state: int = 0,
    n_jobs: int = -1,
    linear_only: bool = False,
) -> dict[str, float]:
    """Train and evaluate a single run."""
    pipeline = create_pipeline(
        n_components=params.get('pca__n_components', 10),
        C=params.get('classifier__estimator__C', 1.0),
        kernel=params.get('classifier__estimator__kernel', 'rbf'),
        gamma=params.get('classifier__estimator__gamma', 'scale'),
        use_balanced=True,
        random_state=random_state,
        n_jobs=n_jobs,
        linear_only=linear_only,
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    results = {
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
    }
    if HAS_IMBALANCED:
        results['geometric_mean'] = geometric_mean_score(y_test, y_pred)

    return results


def run_evaluation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    params: dict,
    n_seeds: int = 3,
    n_jobs: int = -1,
    linear_only: bool = False,
) -> dict[str, tuple[float, float]]:
    """Run evaluation across multiple seeds and return mean ± std."""
    all_results = []
    for seed in range(n_seeds):
        result = evaluate_single_run(
            X_train, y_train, X_test, y_test, params,
            random_state=seed, n_jobs=n_jobs, linear_only=linear_only,
        )
        all_results.append(result)

    # Aggregate results
    metrics = all_results[0].keys()
    summary = {}
    for metric in metrics:
        values = [r[metric] for r in all_results]
        summary[metric] = (np.mean(values), np.std(values))

    return summary


def load_data(csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and split data by subfolder column."""
    df = pd.read_csv(csv_path)

    train_df = df[df['subfolder'] == 'train'].copy()
    val_df = df[df['subfolder'].isin(['val', 'validation'])].copy()
    test_df = df[df['subfolder'] == 'test'].copy()

    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser(description='Benchmark SO(3) invariants for classification')
    parser.add_argument('--input', type=str, required=True, help='Path to CSV with Minkowski tensors')
    parser.add_argument('--spharm-input', type=str, action='append', default=None, help='Path to spherical harmonics CSV (repeatable)')
    parser.add_argument('--spharm-lmax', type=int, nargs='+', default=None, metavar='LMAX',
                        help='lmax values to test (default: lmax from CSV filename). E.g. --spharm-lmax 1 2 3 4 5')
    parser.add_argument('--cellprofiler-input', type=str, default=None, help='Path to CellProfiler features CSV')
    parser.add_argument('--max-so3-degree', type=int, default=3, choices=[1, 2, 3], help='Maximum SO3 polynomial degree to evaluate (default: 3)')
    parser.add_argument('--max-so2-degree', type=int, default=0, choices=[0, 1, 2, 3], help='Maximum SO2 polynomial degree to evaluate (0=disabled, default: 0)')
    parser.add_argument('--include', type=str, nargs='+', default=None, metavar='PATTERN',
                        help='Only run feature sets whose names contain any of these substrings (case-insensitive). E.g. --include "SO2 Degree 1" "SO2 Degree 2"')
    parser.add_argument('--output', type=str, default='benchmarks/results', help='Output directory')
    parser.add_argument('--optimize', action='store_true', help='Run Bayesian optimization')
    parser.add_argument('--n_iter', type=int, default=50, help='Optimization iterations')
    parser.add_argument('--linear-only', action='store_true', help='Restrict SVM kernel search to linear only')
    parser.add_argument('--seeds', type=int, default=3, help='Number of random seeds for evaluation')
    parser.add_argument('--n_jobs', type=int, default=-1, help='Parallel jobs (-1 for all CPUs)')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity level')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    total_start = time.time()

    print("Loading data...")
    train_df, val_df, test_df = load_data(args.input)

    y_train = train_df['label'].values
    y_test = test_df['label'].values

    print(f"Train: {len(train_df)} samples, Test: {len(test_df)} samples")
    print(f"Class distribution (train): {np.bincount(y_train)}")

    # Load spherical harmonics data if provided (one entry per --spharm-input)
    spharm_entries = []  # list of (lmax_val, df)
    for spharm_path in (args.spharm_input or []):
        stem = Path(spharm_path).stem  # e.g. spherical_harmonics_lmax_16
        file_lmax = int(next((p for p in stem.split('_') if p.isdigit()), 5))
        lmax_values = args.spharm_lmax if args.spharm_lmax else [file_lmax]
        print(f"Loading SPHARM from {spharm_path} (testing lmax={lmax_values})...")
        spharm_df = pd.read_csv(spharm_path)
        for lmax_val in lmax_values:
            spharm_entries.append((lmax_val, spharm_df))

    # Define feature sets to evaluate
    feature_sets = [
        ('Minkowski (tensors)', lambda df: build_baseline_features(df, mode='tensors')),
        ('Minkowski (tensors+beta)', lambda df: build_baseline_features(df, mode='tensors+beta')),
        ('Minkowski (tensors+eigen)', lambda df: build_baseline_features(df, mode='tensors+eigen')),
        ('Minkowski (tensors+eigen+beta)', lambda df: build_baseline_features(df, mode='tensors+eigen+beta')),
        ('Beta only', lambda df: build_baseline_features(df, mode='beta')),
        ('Eigenvalues only', lambda df: build_eigen_only_features(df)),
        ('Eigen + Beta', lambda df: build_baseline_features(df, mode='eigen+beta')),
    ]
    for deg in range(1, args.max_so3_degree + 1):
        feature_sets.append(
            (f'SO3 Degree {deg}', lambda df, d=deg: build_invariant_features(df, max_degree=d))
        )

    for deg in range(1, args.max_so3_degree + 1):
        feature_sets.append(
            (f'SO3 Degree {deg} + Eigenvalues', lambda df, d=deg: build_invariants_eigen_features(df, 'SO3', d))
        )

    if args.max_so3_degree >= 2:
        feature_sets.append(
            ('SO3 Degree 2 + SO2 z-scalars', lambda df: build_so3d2_so2d1_extra_features(df))
        )
        feature_sets.append(
            ('SO3 Degree 2 + Eigenvalues', lambda df: build_so3d2_eigen_features(df))
        )

    for deg in range(1, args.max_so2_degree + 1):
        feature_sets.append(
            (f'SO2 Degree {deg}', lambda df, d=deg: build_invariant_features(df, max_degree=d, symmetry='SO2'))
        )
        feature_sets.append(
            (f'SO2 Degree {deg} + Eigenvalues', lambda df, d=deg: build_invariants_eigen_features(df, 'SO2', d))
        )

    if args.cellprofiler_input:
        cp_df = pd.read_csv(args.cellprofiler_input)
        feature_sets.append(
            ('CellProfiler', lambda df, cp=cp_df: build_cellprofiler_features(df, cp))
        )
        feature_sets.append(
            ('CellProfiler (shape only)', lambda df, cp=cp_df: build_cellprofiler_features(df, cp, shape_only=True))
        )

    for lmax_val, spharm_df in spharm_entries:
        feature_sets.append(
            (f'SPHARM lmax={lmax_val}', lambda df, s=spharm_df, l=lmax_val: build_spharm_features(df, s, lmax=l))
        )
        feature_sets.append(
            (f'SPHARM Inv lmax={lmax_val}', lambda df, s=spharm_df, l=lmax_val: build_spharm_invariant_features(df, s, l))
        )

    if args.include:
        patterns = [p.lower() for p in args.include]
        feature_sets = [(n, fn) for n, fn in feature_sets if any(p in n.lower() for p in patterns)]
        if not feature_sets:
            print(f"Error: no feature sets matched --include patterns: {args.include}")
            sys.exit(1)
        print(f"Filtered to {len(feature_sets)} feature set(s): {[n for n, _ in feature_sets]}")

    all_results = {}
    all_hyperparams = {}

    for name, build_fn in feature_sets:
        print(f"\n{'='*60}")
        print(f"Feature set: {name}")
        print('='*60)

        # Build features
        start = time.time()
        X_train, feature_names = build_fn(train_df)
        X_test, _ = build_fn(test_df)
        build_time = time.time() - start

        print(f"  Features: {X_train.shape[1]} ({build_time:.1f}s to compute)")

        # Optimize hyperparameters
        opt_start = time.time()
        if args.optimize:
            print("  Optimizing hyperparameters...")
            params = optimize_hyperparams(
                X_train, y_train,
                n_iter=args.n_iter,
                random_state=0,
                n_jobs=args.n_jobs,
                verbose=args.verbose,
                linear_only=args.linear_only,
            )
        else:
            # Use defaults
            params = {
                'pca__n_components': min(10, X_train.shape[1]),
                'classifier__estimator__C': 1.0,
                'classifier__estimator__kernel': 'rbf',
                'classifier__estimator__gamma': 'scale',
            }
        opt_time = time.time() - opt_start

        print(f"  Params: {params}")
        all_hyperparams[name] = params

        # Evaluate
        print(f"  Evaluating ({args.seeds} seeds)...")
        eval_start = time.time()
        results = run_evaluation(
            X_train, y_train, X_test, y_test, params,
            n_seeds=args.seeds, n_jobs=args.n_jobs, linear_only=args.linear_only,
        )
        eval_time = time.time() - eval_start

        all_results[name] = {
            'n_features': X_train.shape[1],
            'feature_names': feature_names if len(feature_names) <= 50 else f'{len(feature_names)} features',
            **{k: {'mean': v[0], 'std': v[1]} for k, v in results.items()},
            'runtime_seconds': {'optimization': round(opt_time, 1), 'evaluation': round(eval_time, 1)},
        }

        print(f"  Results:")
        for metric, (mean, std) in results.items():
            print(f"    {metric}: {mean:.3f} ± {std:.3f}")

    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Feature Set':<25} | {'# Feat':>7} | {'Bal. Acc':>15} | {'Geo. Mean':>15}")
    print("-"*80)
    for name, res in all_results.items():
        bal_acc = res.get('balanced_accuracy', {})
        geo_mean = res.get('geometric_mean', {})
        bal_str = f"{bal_acc.get('mean', 0):.3f} ± {bal_acc.get('std', 0):.3f}" if bal_acc else "N/A"
        geo_str = f"{geo_mean.get('mean', 0):.3f} ± {geo_mean.get('std', 0):.3f}" if geo_mean else "N/A"
        print(f"{name:<25} | {res['n_features']:>7} | {bal_str:>15} | {geo_str:>15}")
    print("="*80)

    # Save results - derive dataset name from output directory or input file
    dataset_name = os.path.basename(args.output.rstrip('/')) or \
                   os.path.basename(os.path.dirname(args.input))

    total_elapsed = time.time() - total_start
    total_h = int(total_elapsed // 3600)
    total_m = int((total_elapsed % 3600) // 60)
    total_s = int(total_elapsed % 60)
    print(f"\nTotal runtime: {total_h}h {total_m}m {total_s}s ({total_elapsed:.0f}s)")

    results_path = os.path.join(args.output, f'{dataset_name}_invariants_scores.json')
    # Attach total runtime to saved results
    all_results['_meta'] = {
        'total_runtime_seconds': round(total_elapsed, 1),
        'n_iter': args.n_iter,
        'seeds': args.seeds,
        'linear_only': args.linear_only,
    }
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to: {results_path}")

    hyperparams_path = os.path.join(args.output, f'{dataset_name}_invariants_hyperparams.json')
    with open(hyperparams_path, 'w') as f:
        json.dump(all_hyperparams, f, indent=2, default=str)
    print(f"Hyperparams saved to: {hyperparams_path}")


if __name__ == '__main__':
    main()
