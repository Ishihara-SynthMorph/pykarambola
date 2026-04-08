"""
Rotation-invariant features from SPHARM (Spherical Harmonic) coefficients.

Implements power spectrum and bispectrum extraction from aics-shparam CSV output.
Both are provably SO(3)-invariant, unlike raw SH coefficients.

References:
  Kazhdan et al. (2003) "Rotation Invariant Spherical Harmonic Representation of 3D Shape Descriptors"
  Kondor (2007) "A novel set of rotationally and translationally invariant features for images"
"""

from __future__ import annotations

import math
from functools import lru_cache
from itertools import product

import numpy as np
import pandas as pd

from pykarambola.spherical import _wigner3j


@lru_cache(maxsize=None)
def _cg(l1: int, m1: int, l2: int, m2: int, l: int, m: int) -> float:
    """Clebsch-Gordan coefficient <l1,m1; l2,m2 | l,m>.

    Computed via Wigner 3j symbol:
      CG(l1,m1; l2,m2 | l,m) = (-1)^(l1-l2+m) * sqrt(2l+1) * W3j(l1,l2,l; m1,m2,-m)
    """
    if m1 + m2 != m:
        return 0.0
    return (-1) ** (l1 - l2 + m) * math.sqrt(2 * l + 1) * _wigner3j(l1, l2, l, m1, m2, -m)


def parse_spharm_df(df: pd.DataFrame, lmax: int) -> np.ndarray:
    """Extract complex SH coefficients from aics-shparam CSV format.

    Columns follow the naming convention: shcoeffs_L{l}M{m}C (cosine) and
    shcoeffs_L{l}M{m}S (sine) for l=0..lmax, m=0..lmax.
    Pairs with m>l are zero-padded and are ignored here.

    Returns array of shape (n_samples, lmax+1, 2*lmax+1) where
    index [n, l, m+lmax] stores f_{l,m} as a complex number.

    Real-to-complex convention (consistent with physics convention):
      f_{l,0}  = c_{l,0}                         (m=0)
      f_{l,+m} = (-1)^m / sqrt(2) * (c - i*s)   (m>0)
      f_{l,-m} = 1/sqrt(2) * (c + i*s)           (m>0)
    """
    n = len(df)
    f = np.zeros((n, lmax + 1, 2 * lmax + 1), dtype=complex)

    for l in range(lmax + 1):
        # m=0: cosine only
        col_c = f'shcoeffs_L{l}M0C'
        if col_c in df.columns:
            f[:, l, lmax] = df[col_c].values  # m=0 stored at index lmax

        for m in range(1, l + 1):
            col_c = f'shcoeffs_L{l}M{m}C'
            col_s = f'shcoeffs_L{l}M{m}S'
            if col_c not in df.columns or col_s not in df.columns:
                continue
            c = df[col_c].values
            s = df[col_s].values
            sign = (-1) ** m
            inv_sqrt2 = 1.0 / math.sqrt(2)
            f[:, l, lmax + m] = sign * inv_sqrt2 * (c - 1j * s)   # f_{l,+m}
            f[:, l, lmax - m] = inv_sqrt2 * (c + 1j * s)           # f_{l,-m}

    return f


def power_spectrum(f_lm: np.ndarray, lmax: int) -> np.ndarray:
    """Compute rotation-invariant power spectrum S_l = Σ_m |f_{l,m}|².

    Parameters
    ----------
    f_lm : ndarray, shape (n_samples, lmax+1, 2*lmax+1)
        Complex SH coefficients as returned by parse_spharm_df.
    lmax : int
        Maximum spherical harmonic degree.

    Returns
    -------
    S : ndarray, shape (n_samples, lmax+1)
    """
    n = f_lm.shape[0]
    S = np.zeros((n, lmax + 1))
    for l in range(lmax + 1):
        # Only sum over valid m: -l..l
        m_slice = slice(lmax - l, lmax + l + 1)
        S[:, l] = np.sum(np.abs(f_lm[:, l, m_slice]) ** 2, axis=1)
    return S


def _valid_bispectrum_triples(lmax: int) -> list[tuple[int, int, int]]:
    """Return sorted list of valid (l1, l2, l) triples for bispectrum.

    Valid: 0 <= l1 <= l2 <= lmax, |l1-l2| <= l <= min(l1+l2, lmax).
    """
    triples = []
    for l1 in range(lmax + 1):
        for l2 in range(l1, lmax + 1):
            for l in range(abs(l1 - l2), min(l1 + l2, lmax) + 1):
                triples.append((l1, l2, l))
    return triples


def bispectrum(f_lm: np.ndarray, lmax: int) -> np.ndarray:
    """Compute rotation-invariant bispectrum features.

    B_{l1,l2,l} = Re( Σ_{m1,m2} CG(l1,m1; l2,m2 | l,m1+m2) · f_{l1,m1} · f_{l2,m2} · conj(f_{l,m1+m2}) )

    Parameters
    ----------
    f_lm : ndarray, shape (n_samples, lmax+1, 2*lmax+1)
    lmax : int

    Returns
    -------
    B : ndarray, shape (n_samples, n_triples)
        Real-valued bispectrum features. n_triples = 69 for lmax=5.
    """
    triples = _valid_bispectrum_triples(lmax)
    n = f_lm.shape[0]
    B = np.zeros((n, len(triples)))

    for idx, (l1, l2, l) in enumerate(triples):
        # Precompute nonzero CG entries for this triple
        cg_entries = []
        for m1 in range(-l1, l1 + 1):
            for m2 in range(-l2, l2 + 1):
                m = m1 + m2
                if abs(m) > l:
                    continue
                cg_val = _cg(l1, m1, l2, m2, l, m)
                if cg_val != 0.0:
                    cg_entries.append((cg_val, m1, m2, m))

        # Sum over nonzero CG entries (vectorised over batch)
        acc = np.zeros(n, dtype=complex)
        for cg_val, m1, m2, m in cg_entries:
            acc += cg_val * (
                f_lm[:, l1, lmax + m1]
                * f_lm[:, l2, lmax + m2]
                * np.conj(f_lm[:, l, lmax + m])
            )
        B[:, idx] = acc.real

    return B


def _bispectrum_feature_names(lmax: int) -> list[str]:
    return [f'bispec_{l1}_{l2}_{l}' for l1, l2, l in _valid_bispectrum_triples(lmax)]


def compute_spharm_invariants(
    df: pd.DataFrame,
    lmax: int = 5,
    include_bispectrum: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Compute rotation-invariant SPHARM features from aics-shparam CSV data.

    Parameters
    ----------
    df : DataFrame
        Must contain shcoeffs_L{l}M{m}C/S columns (aics-shparam convention).
    lmax : int
        Maximum spherical harmonic degree (default 5).
    include_bispectrum : bool
        If True (default), include bispectrum in addition to power spectrum.

    Returns
    -------
    X : ndarray, shape (n_samples, n_features)
        n_features = (lmax+1) + n_bispec_triples if include_bispectrum,
        else (lmax+1). For lmax=5: 6 + 69 = 75.
    feature_names : list[str]
        Names for each column of X.
    """
    f_lm = parse_spharm_df(df, lmax)

    S = power_spectrum(f_lm, lmax)
    ps_names = [f'power_l{l}' for l in range(lmax + 1)]

    if include_bispectrum:
        B = bispectrum(f_lm, lmax)
        bs_names = _bispectrum_feature_names(lmax)
        X = np.concatenate([S, B], axis=1)
        feature_names = ps_names + bs_names
    else:
        X = S
        feature_names = ps_names

    return X, feature_names
