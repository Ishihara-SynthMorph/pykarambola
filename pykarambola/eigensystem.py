# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Yajushi Khurana and Keisuke Ishihara
# This file is part of pykarambola, a Python port of karambola.
# See LICENSE for the full license text.
"""
Eigenvalue/eigenvector decomposition for symmetric 3x3 matrices.
"""

import numpy as np
from .results import MinkValResult


class EigenSystem:
    """Stores eigenvalues and eigenvectors."""
    def __init__(self):
        self.eigen_values = [0.0, 0.0, 0.0]
        self.eigen_vectors = [[0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0]]


def calculate_eigensystem(w_matrix_results):
    """Compute eigenvalues and eigenvectors for each label in a matrix result.

    Parameters
    ----------
    w_matrix_results : dict[int, MinkValResult]
        Matrix results keyed by label.

    Returns
    -------
    dict[int, MinkValResult]
        Eigensystem results keyed by label.
    """
    eigsys_results = {}

    for label, w_labeled in w_matrix_results.items():
        eigsys = MinkValResult(result=EigenSystem())
        eigsys.name = w_labeled.name + "_eigsys"
        for kw in w_labeled.keywords:
            eigsys.append_keyword(kw)

        # Check for NaN in matrix
        mat = w_labeled.result.to_numpy()
        calc_eigsys = not np.any(np.isnan(mat))

        if calc_eigsys:
            # Use numpy's eigh for symmetric matrices
            eigenvalues, eigenvectors = np.linalg.eigh(mat)

            # Sort by ascending magnitude (|λ₁| ≤ |λ₂| ≤ |λ₃|), matching C++ Karambola
            order = np.argsort(np.abs(eigenvalues))
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]

            for i in range(3):
                eigsys.result.eigen_values[i] = eigenvalues[i]
                evec = eigenvectors[:, i]
                # Flip eigenvector so z-component is non-negative
                if evec[2] < 0:
                    evec = -evec
                eigsys.result.eigen_vectors[i] = [evec[0], evec[1], evec[2]]
        else:
            for i in range(3):
                eigsys.result.eigen_values[i] = float('nan')
                eigsys.result.eigen_vectors[i] = [float('nan'), float('nan'), float('nan')]

        eigsys_results[label] = eigsys

    return eigsys_results
