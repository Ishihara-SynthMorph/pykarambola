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
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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


def build_baseline_features(df: pd.DataFrame, include_eigen: bool = False) -> tuple[np.ndarray, list[str]]:
    """Extract baseline raw tensor features.

    Matches minkowski_classifier approach: use all columns from index 3+,
    optionally filtering out beta/EVal columns.
    """
    # Get all feature columns (skip image_num, label, subfolder)
    all_feature_cols = df.columns[3:].tolist()

    if include_eigen:
        # tensors_with_eigen_values: use all columns
        cols = all_feature_cols
    else:
        # tensors: filter out beta and EVal columns
        cols = [c for c in all_feature_cols if 'beta' not in c and 'EVal' not in c]

    X = df[cols].values
    return X, cols


def create_pipeline(n_components: int, C: float, kernel: str, gamma: str,
                    use_balanced: bool = True, random_state: int = 0, n_jobs: int = -1):
    """Create classification pipeline."""
    svc = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)

    if use_balanced and HAS_IMBALANCED:
        classifier = BalancedBaggingClassifier(
            estimator=svc,
            sampling_strategy='auto',
            replacement=False,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    else:
        from sklearn.ensemble import BaggingClassifier
        classifier = BaggingClassifier(
            estimator=svc,
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

    search_space = {
        'pca__n_components': Integer(2, min(X_train.shape[1], X_train.shape[0] - 1)),
        'classifier__estimator__C': Real(1e-1, 1e3, prior='log-uniform'),
        'classifier__estimator__kernel': Categorical(['linear', 'rbf']),
        'classifier__estimator__gamma': Categorical(['scale', 'auto']),
    }

    # Create pipeline template
    svc = SVC()
    if HAS_IMBALANCED:
        classifier = BalancedBaggingClassifier(
            estimator=svc,
            sampling_strategy='auto',
            replacement=False,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    else:
        from sklearn.ensemble import BaggingClassifier
        classifier = BaggingClassifier(estimator=svc, random_state=random_state, n_jobs=n_jobs)

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
) -> dict[str, tuple[float, float]]:
    """Run evaluation across multiple seeds and return mean ± std."""
    all_results = []
    for seed in range(n_seeds):
        result = evaluate_single_run(
            X_train, y_train, X_test, y_test, params,
            random_state=seed, n_jobs=n_jobs
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
    parser.add_argument('--output', type=str, default='benchmarks/results', help='Output directory')
    parser.add_argument('--optimize', action='store_true', help='Run Bayesian optimization')
    parser.add_argument('--n_iter', type=int, default=50, help='Optimization iterations')
    parser.add_argument('--seeds', type=int, default=3, help='Number of random seeds for evaluation')
    parser.add_argument('--n_jobs', type=int, default=-1, help='Parallel jobs (-1 for all CPUs)')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity level')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Loading data...")
    train_df, val_df, test_df = load_data(args.input)

    y_train = train_df['label'].values
    y_test = test_df['label'].values

    print(f"Train: {len(train_df)} samples, Test: {len(test_df)} samples")
    print(f"Class distribution (train): {np.bincount(y_train)}")

    # Define feature sets to evaluate
    feature_sets = [
        ('Baseline (tensors)', lambda df: build_baseline_features(df, include_eigen=False)),
        ('Baseline (w/ eigen)', lambda df: build_baseline_features(df, include_eigen=True)),
        ('SO3 Degree 1', lambda df: build_invariant_features(df, max_degree=1)),
        ('SO3 Degree 2', lambda df: build_invariant_features(df, max_degree=2)),
        ('SO3 Degree 3', lambda df: build_invariant_features(df, max_degree=3)),
    ]

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
        if args.optimize:
            print("  Optimizing hyperparameters...")
            params = optimize_hyperparams(
                X_train, y_train,
                n_iter=args.n_iter,
                random_state=0,
                n_jobs=args.n_jobs,
                verbose=args.verbose,
            )
        else:
            # Use defaults
            params = {
                'pca__n_components': min(10, X_train.shape[1]),
                'classifier__estimator__C': 1.0,
                'classifier__estimator__kernel': 'rbf',
                'classifier__estimator__gamma': 'scale',
            }

        print(f"  Params: {params}")
        all_hyperparams[name] = params

        # Evaluate
        print(f"  Evaluating ({args.seeds} seeds)...")
        results = run_evaluation(
            X_train, y_train, X_test, y_test, params,
            n_seeds=args.seeds, n_jobs=args.n_jobs
        )

        all_results[name] = {
            'n_features': X_train.shape[1],
            'feature_names': feature_names if len(feature_names) <= 50 else f'{len(feature_names)} features',
            **{k: {'mean': v[0], 'std': v[1]} for k, v in results.items()},
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

    results_path = os.path.join(args.output, f'{dataset_name}_invariants_scores.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")

    hyperparams_path = os.path.join(args.output, f'{dataset_name}_invariants_hyperparams.json')
    with open(hyperparams_path, 'w') as f:
        json.dump(all_hyperparams, f, indent=2, default=str)
    print(f"Hyperparams saved to: {hyperparams_path}")


if __name__ == '__main__':
    main()
