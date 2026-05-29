"""
Compatibility tests for the aicsshparam.shinvariants upstream shim (issue #137).

All tests are skipped when aicsshparam is not installed.  Once the upstream PR
is merged and the package is released to PyPI, all tests should pass green with
no further code changes in spharm_invariants.py.

Expected feature counts per lmax (power spectrum + bispectrum triples):
  lmax=1:  2 + 4  =  6
  lmax=2:  3 + 11 = 14
  lmax=3:  4 + 23 = 27
  lmax=5:  6 + 69 = 75
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

shinvariants = pytest.importorskip("aicsshparam.shinvariants")

from pykarambola.spharm_invariants import (  # noqa: E402 — after importorskip
    _HAS_UPSTREAM,
    _valid_bispectrum_triples,
    compute_spharm_invariants,
    parse_spharm_df,
    power_spectrum,
)


def _make_toy_df(n: int, lmax: int, rng: np.random.Generator) -> pd.DataFrame:
    data: dict = {}
    for l in range(lmax + 1):
        data[f"shcoeffs_L{l}M0C"] = rng.standard_normal(n)
        for m in range(1, lmax + 1):
            data[f"shcoeffs_L{l}M{m}C"] = rng.standard_normal(n)
            data[f"shcoeffs_L{l}M{m}S"] = rng.standard_normal(n)
    return pd.DataFrame(data)


class TestShimRouting:
    def test_upstream_detected(self):
        assert _HAS_UPSTREAM, "_HAS_UPSTREAM should be True when aicsshparam is installed"

    def test_compute_routes_to_upstream(self):
        """compute_spharm_invariants must return identical results to get_invariants."""
        rng = np.random.default_rng(0)
        lmax = 3
        df = _make_toy_df(5, lmax, rng)
        X_shim, names_shim = compute_spharm_invariants(df, lmax=lmax, include_bispectrum=True)
        X_up, names_up = shinvariants.get_invariants(df, lmax=lmax)
        np.testing.assert_array_equal(X_shim, X_up)
        assert names_shim == names_up

    def test_power_only_still_uses_local(self):
        """include_bispectrum=False must not route to upstream (upstream has no such mode)."""
        rng = np.random.default_rng(1)
        lmax = 3
        df = _make_toy_df(4, lmax, rng)
        X, names = compute_spharm_invariants(df, lmax=lmax, include_bispectrum=False)
        assert X.shape == (4, lmax + 1)
        assert len(names) == lmax + 1


@pytest.mark.parametrize("lmax,n_expected", [
    (1,  6),
    (2, 14),
    (3, 27),
    (5, 75),
])
def test_upstream_feature_count(lmax, n_expected):
    rng = np.random.default_rng(0)
    df = _make_toy_df(4, lmax, rng)
    X, names = shinvariants.get_invariants(df, lmax=lmax)
    assert X.shape == (4, n_expected), (
        f"lmax={lmax}: expected (4, {n_expected}), got {X.shape}"
    )
    assert len(names) == n_expected


class TestPowerSpectrumAgreement:
    """Power spectrum features must agree between upstream and local implementation.

    Bispectrum features will differ for odd-J triples (l1+l2+l odd) — this is
    expected and documented in issue #137: the upstream uses the correct Racah
    Wigner 3j while the local implementation zeros those terms.
    """

    def test_power_spectrum_values_match(self):
        rng = np.random.default_rng(42)
        lmax = 3
        df = _make_toy_df(6, lmax, rng)

        X_up, names_up = shinvariants.get_invariants(df, lmax=lmax)

        ps_indices = [i for i, n in enumerate(names_up) if n.startswith("power_")]
        if not ps_indices:
            pytest.skip(
                "Upstream feature names do not use 'power_' prefix — "
                "update ps_indices selection to match upstream naming"
            )

        X_ps_up = X_up[:, ps_indices]
        f_lm = parse_spharm_df(df, lmax)
        S_local = power_spectrum(f_lm, lmax)

        np.testing.assert_allclose(
            X_ps_up, S_local, atol=1e-10,
            err_msg="Power spectrum disagrees between upstream and local",
        )

    def test_even_j_bispectrum_values_match(self):
        """Bispectrum triples with even l1+l2+l must agree; odd-J triples may differ."""
        rng = np.random.default_rng(99)
        lmax = 3
        df = _make_toy_df(4, lmax, rng)

        X_up, names_up = shinvariants.get_invariants(df, lmax=lmax)

        bs_names_up = [n for n in names_up if n.startswith("bispec_")]
        if not bs_names_up:
            pytest.skip(
                "Upstream feature names do not use 'bispec_' prefix — "
                "update selection to match upstream naming"
            )

        # Identify even-J triples in the upstream output
        even_j_local = {
            (l1, l2, l)
            for l1, l2, l in _valid_bispectrum_triples(lmax)
            if (l1 + l2 + l) % 2 == 0
        }

        # Indices of even-J features in upstream output (assumes bispec_l1_l2_l naming)
        import re
        even_up_indices, even_local_indices = [], []
        local_triples = _valid_bispectrum_triples(lmax)
        for up_i, name in enumerate(names_up):
            m = re.fullmatch(r"bispec_(\d+)_(\d+)_(\d+)", name)
            if m:
                triple = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
                if triple in even_j_local:
                    even_up_indices.append(up_i)
                    even_local_indices.append(local_triples.index(triple))

        if not even_up_indices:
            pytest.skip("Could not match any even-J bispectrum triples by name")

        from pykarambola.spharm_invariants import bispectrum
        f_lm = parse_spharm_df(df, lmax)
        B_local = bispectrum(f_lm, lmax)

        np.testing.assert_allclose(
            X_up[:, even_up_indices], B_local[:, even_local_indices],
            atol=1e-8,
            err_msg="Even-J bispectrum features disagree between upstream and local",
        )
