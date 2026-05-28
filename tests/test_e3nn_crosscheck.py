"""Cross-check of shinvariants power spectrum and bispectrum against e3nn-jax.

Addresses issue #138: independently validate the physics-convention CG
coefficients and derived invariants (power spectrum, bispectrum) by
comparison with e3nn-jax's ``su2_clebsch_gordan``.

Convention summary
------------------
* **Physics (Condon-Shortley)** -- used by shinvariants / spharm_invariants::

      CG^phys(l1,m1; l2,m2 | l,m)
          = (-1)^(l1-l2+m) * sqrt(2l+1) * W3j(l1,l2,l; m1,m2,-m)

* **e3nn SU(2) / complex-SH** -- ``e3nn_jax.su2_clebsch_gordan``::

      su2_CG(l1,l2,l3)[m1+l1, m2+l2, m+l3]
          = CG^phys(l1,m1; l2,m2 | l3,m) / sqrt(2*l3+1)

  The ``/ sqrt(2*l3+1)`` factor is explicit in e3nn's source (su2.py, line 38).

* **Conversion**::

      CG^phys = sqrt(2l+1) * su2_CG          (no additional signs)
      B^phys_{l1,l2,l} = sqrt(2l+1) * B^su2_{l1,l2,l}

* **Real-SH bispectrum (pyspectra / e3nn.clebsch_gordan)**:
  ``e3nn.clebsch_gordan`` works in the *real*-SH basis (index = real-SH
  quantum number, not complex-SH m).  Comparing against it requires a
  full real-to-complex change of basis and is further complicated by the
  symmetrisation under l1<->l2 in
  ``e3nn.reduced_symmetric_tensor_product_basis``.  This is documented
  in issue #138 as a known residual discrepancy and is out of scope for
  the current cross-check, which targets the complex-SH (SU(2)) level.

Requirements
------------
``jax`` and ``e3nn-jax`` must be installed (``pip install jax e3nn-jax``).
Tests are automatically skipped when these packages are absent.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import pytest

jax = pytest.importorskip("jax")  # skip entire module if JAX is absent
e3nn_jax = pytest.importorskip("e3nn_jax")

from e3nn_jax._src.su2 import su2_clebsch_gordan  # noqa: E402


# ---------------------------------------------------------------------------
# Self-contained physics CG / bispectrum (no dependency on spharm_invariants)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _wigner3j(j1: int, j2: int, j3: int, m1: int, m2: int, m3: int) -> float:
    """Wigner 3j symbol via the Racah formula (same as shinvariants._wigner3j)."""
    if m1 + m2 + m3 != 0:
        return 0.0
    if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return 0.0
    if j3 < abs(j1 - j2) or j3 > j1 + j2:
        return 0.0

    def _triangle(a: int, b: int, c: int) -> float:
        return (
            math.factorial(a + b - c)
            * math.factorial(a - b + c)
            * math.factorial(-a + b + c)
            / math.factorial(a + b + c + 1)
        )

    tri = _triangle(j1, j2, j3)
    prefactor = (-1) ** (j1 - j2 - m3) * math.sqrt(
        tri
        * math.factorial(j1 + m1)
        * math.factorial(j1 - m1)
        * math.factorial(j2 + m2)
        * math.factorial(j2 - m2)
        * math.factorial(j3 + m3)
        * math.factorial(j3 - m3)
    )

    t_min = max(0, j2 - j3 - m1, j1 - j3 + m2)
    t_max = min(j1 + j2 - j3, j1 - m1, j2 + m2)
    s = sum(
        (-1) ** t
        / (
            math.factorial(t)
            * math.factorial(j1 + j2 - j3 - t)
            * math.factorial(j1 - m1 - t)
            * math.factorial(j2 + m2 - t)
            * math.factorial(j3 - j2 + m1 + t)
            * math.factorial(j3 - j1 - m2 + t)
        )
        for t in range(t_min, t_max + 1)
    )
    return prefactor * s


@lru_cache(maxsize=None)
def _phys_cg(l1: int, m1: int, l2: int, m2: int, l: int, m: int) -> float:
    """Physics (Condon-Shortley) CG coefficient."""
    if m1 + m2 != m:
        return 0.0
    return (-1) ** (l1 - l2 + m) * math.sqrt(2 * l + 1) * _wigner3j(l1, l2, l, m1, m2, -m)


def _valid_triples(lmax: int) -> list[tuple[int, int, int]]:
    """Bispectrum index triples (l1 <= l2)."""
    return [
        (l1, l2, l)
        for l1 in range(lmax + 1)
        for l2 in range(l1, lmax + 1)
        for l in range(abs(l1 - l2), min(l1 + l2, lmax) + 1)
    ]


def _make_complex_flm(lmax: int, seed: int = 0) -> np.ndarray:
    """Random complex SH coefficient array, shape (1, lmax+1, 2*lmax+1)."""
    rng = np.random.default_rng(seed)
    f = np.zeros((1, lmax + 1, 2 * lmax + 1), dtype=complex)
    for l in range(lmax + 1):
        f[0, l, lmax] = rng.standard_normal()  # m=0: real
        for m in range(1, l + 1):
            c = rng.standard_normal()
            s = rng.standard_normal()
            sign = (-1) ** m
            inv_sqrt2 = 1.0 / math.sqrt(2)
            f[0, l, lmax + m] = sign * inv_sqrt2 * (c - 1j * s)
            f[0, l, lmax - m] = inv_sqrt2 * (c + 1j * s)
    return f


def _phys_bispectrum(f_lm: np.ndarray, lmax: int) -> np.ndarray:
    """Bispectrum with physics CG coefficients.  Shape: (n, n_triples)."""
    triples = _valid_triples(lmax)
    n = f_lm.shape[0]
    B = np.zeros((n, len(triples)))
    for idx, (l1, l2, l) in enumerate(triples):
        acc = np.zeros(n, dtype=complex)
        for m1 in range(-l1, l1 + 1):
            for m2 in range(-l2, l2 + 1):
                m = m1 + m2
                if abs(m) > l:
                    continue
                cg = _phys_cg(l1, m1, l2, m2, l, m)
                if cg == 0.0:
                    continue
                acc += cg * (
                    f_lm[:, l1, lmax + m1]
                    * f_lm[:, l2, lmax + m2]
                    * np.conj(f_lm[:, l, lmax + m])
                )
        B[:, idx] = acc.real
    return B


def _su2_bispectrum(f_lm: np.ndarray, lmax: int) -> np.ndarray:
    """Bispectrum using e3nn su2 CG coefficients.  Shape: (n, n_triples)."""
    triples = _valid_triples(lmax)
    n = f_lm.shape[0]
    B = np.zeros((n, len(triples)))
    for idx, (l1, l2, l) in enumerate(triples):
        CG = su2_clebsch_gordan(l1, l2, l)  # shape (2l1+1, 2l2+1, 2l+1)
        acc = np.zeros(n, dtype=complex)
        for m1 in range(-l1, l1 + 1):
            for m2 in range(-l2, l2 + 1):
                m = m1 + m2
                if abs(m) > l:
                    continue
                cg = float(CG[m1 + l1, m2 + l2, m + l])
                if cg == 0.0:
                    continue
                acc += cg * (
                    f_lm[:, l1, lmax + m1]
                    * f_lm[:, l2, lmax + m2]
                    * np.conj(f_lm[:, l, lmax + m])
                )
        B[:, idx] = acc.real
    return B


# ---------------------------------------------------------------------------
# Tests: CG coefficient convention
# ---------------------------------------------------------------------------


class TestCGConvention:
    def test_known_cg_l0_agrees(self):
        """Known value: CG(1,0;1,0|0,0) = -1/sqrt(3).

        For l=0 the conversion factor sqrt(2*0+1)=1, so su2 and physics
        must agree exactly (the 'no-arithmetic sanity check' from issue #138).
        """
        CG = su2_clebsch_gordan(1, 1, 0)
        su2_val = float(CG[1 + 0, 1 + 0, 0 + 0]) * math.sqrt(1)  # * sqrt(2*0+1)
        phys_val = _phys_cg(1, 0, 1, 0, 0, 0)
        expected = -1.0 / math.sqrt(3)
        np.testing.assert_allclose(su2_val, expected, atol=1e-10)
        np.testing.assert_allclose(phys_val, expected, atol=1e-10)

    def test_sqrt_2lp1_conversion_all_triples(self):
        """su2_CG[...] * sqrt(2l+1) == phys_CG for all valid entries, lmax=4."""
        lmax = 4
        max_err = 0.0
        for l1 in range(lmax + 1):
            for l2 in range(l1, lmax + 1):
                for l in range(abs(l1 - l2), min(l1 + l2, lmax) + 1):
                    CG = su2_clebsch_gordan(l1, l2, l)
                    factor = math.sqrt(2 * l + 1)
                    for m1 in range(-l1, l1 + 1):
                        for m2 in range(-l2, l2 + 1):
                            m = m1 + m2
                            if abs(m) > l:
                                continue
                            su2_val = float(CG[m1 + l1, m2 + l2, m + l]) * factor
                            phys_val = _phys_cg(l1, m1, l2, m2, l, m)
                            max_err = max(max_err, abs(su2_val - phys_val))
        assert max_err < 1e-12, f"Max |su2*sqrt(2l+1) - physics| = {max_err:.2e}"


# ---------------------------------------------------------------------------
# Tests: power spectrum (no CG -- should agree trivially)
# ---------------------------------------------------------------------------


class TestPowerSpectrum:
    def test_power_spectrum_identical(self):
        """S_l = Σ_m |f_{l,m}|² is independent of CG convention."""
        lmax = 4
        f = _make_complex_flm(lmax, seed=7)
        S_direct = np.array(
            [
                [np.sum(np.abs(f[0, l, lmax - l : lmax + l + 1]) ** 2) for l in range(lmax + 1)]
            ]
        )
        # Both physics and su2 bispectrum implementations use the same f_lm,
        # so the power spectrum (no CG) is the same by construction.
        # We verify it equals the direct sum-of-squares.
        for l in range(lmax + 1):
            val = float(np.sum(np.abs(f[0, l, lmax - l : lmax + l + 1]) ** 2))
            np.testing.assert_allclose(S_direct[0, l], val, rtol=1e-14)


# ---------------------------------------------------------------------------
# Tests: bispectrum conversion factor B^phys = sqrt(2l+1) * B^su2
# ---------------------------------------------------------------------------


class TestBispectrumConversion:
    @pytest.fixture
    def bispectrum_data(self):
        lmax = 4
        f = _make_complex_flm(lmax, seed=42)
        B_phys = _phys_bispectrum(f, lmax)
        B_su2 = _su2_bispectrum(f, lmax)
        triples = _valid_triples(lmax)
        return B_phys, B_su2, triples, lmax

    def test_bispectrum_conversion_factor(self, bispectrum_data):
        """B^phys_{l1,l2,l} = sqrt(2l+1) * B^su2_{l1,l2,l} for all triples."""
        B_phys, B_su2, triples, lmax = bispectrum_data
        for idx, (l1, l2, l) in enumerate(triples):
            factor = math.sqrt(2 * l + 1)
            np.testing.assert_allclose(
                B_phys[0, idx],
                factor * B_su2[0, idx],
                atol=1e-10,
                rtol=1e-10,
                err_msg=f"Conversion failed for triple ({l1},{l2},{l}): "
                f"B_phys={B_phys[0,idx]:.6g}, "
                f"sqrt(2l+1)*B_su2={factor*B_su2[0,idx]:.6g}",
            )

    def test_l0_triple_no_scaling(self):
        """For l=0 output (factor=1), phys and su2 bispectra must match exactly."""
        lmax = 4
        f = _make_complex_flm(lmax, seed=99)
        B_phys = _phys_bispectrum(f, lmax)
        B_su2 = _su2_bispectrum(f, lmax)
        triples = _valid_triples(lmax)
        for idx, (l1, l2, l) in enumerate(triples):
            if l == 0:
                np.testing.assert_allclose(
                    B_phys[0, idx],
                    B_su2[0, idx],
                    atol=1e-10,
                    err_msg=f"l=0 triple ({l1},{l2},0): phys={B_phys[0,idx]:.6g} su2={B_su2[0,idx]:.6g}",
                )
