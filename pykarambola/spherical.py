"""
Spherical Minkowski tensors using spherical harmonics.
"""

import math
from functools import lru_cache

import numpy as np
try:
    from scipy.special import sph_harm_y
    def _sph_harm(m, l, phi, theta):
        """Wrapper matching old sph_harm(m, l, phi, theta) convention."""
        return sph_harm_y(l, m, theta, phi)
except ImportError:
    from scipy.special import sph_harm as _sph_harm
from .results import MinkValResult

MAX_L = 8


@lru_cache(maxsize=None)
def _wigner3j(j1, j2, j3, m1, m2, m3):
    """Wigner 3j symbol via the Racah formula.

    All arguments must be integers (not half-integers).
    Sufficient for l <= 8 used in spherical Minkowski tensors.
    """
    # Selection rules
    if m1 + m2 + m3 != 0:
        return 0.0
    if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return 0.0
    if j3 < abs(j1 - j2) or j3 > j1 + j2:
        return 0.0
    J = j1 + j2 + j3
    if J % 2 != 0:
        return 0.0

    # Triangle coefficient
    def _triangle(a, b, c):
        return (math.factorial(a + b - c)
                * math.factorial(a - b + c)
                * math.factorial(-a + b + c)
                / math.factorial(a + b + c + 1))

    tri = _triangle(j1, j2, j3)
    prefactor = ((-1) ** (j1 - j2 - m3)
                 * math.sqrt(tri
                             * math.factorial(j1 + m1)
                             * math.factorial(j1 - m1)
                             * math.factorial(j2 + m2)
                             * math.factorial(j2 - m2)
                             * math.factorial(j3 + m3)
                             * math.factorial(j3 - m3)))

    # Sum over t
    t_min = max(0, j2 - j3 - m1, j1 - j3 + m2)
    t_max = min(j1 + j2 - j3, j1 - m1, j2 + m2)
    s = 0.0
    for t in range(t_min, t_max + 1):
        s += ((-1) ** t
              / (math.factorial(t)
                 * math.factorial(j1 + j2 - j3 - t)
                 * math.factorial(j1 - m1 - t)
                 * math.factorial(j2 + m2 - t)
                 * math.factorial(j3 - j2 + m1 + t)
                 * math.factorial(j3 - j1 - m2 + t)))

    return prefactor * s


class SphMinkData:
    """Container for ql and wl arrays."""
    def __init__(self):
        self.ql = [0.0] * 12
        self.wl = [0.0] * 12


def _default_sphmink():
    return MinkValResult(result=SphMinkData())


class SphericalMinkowskis:
    """Accumulates spherical harmonic coefficients for Minkowski tensors."""

    def __init__(self):
        # Store d_lm coefficients: for each l, store m = 0..l as complex
        self._d = {}  # (l, m) -> complex
        for l in range(MAX_L + 1):
            for m in range(l + 1):
                self._d[(l, m)] = 0.0 + 0.0j
        self.total_area = 0.0

    def add_facets_batch(self, normals, areas):
        """Add contributions from multiple facets at once.

        Parameters
        ----------
        normals : (N, 3) array — unit normals
        areas : (N,) array — triangle areas
        """
        f = normals * areas[:, None]  # (N, 3)
        f_norms = np.linalg.norm(f, axis=1)
        safe_norms = np.where(f_norms > 0, f_norms, 1.0)
        f_normalized = f / safe_norms[:, None]

        cos_th = f_normalized[:, 2]
        phi = np.arctan2(f_normalized[:, 1], f_normalized[:, 0])
        theta = np.arccos(np.clip(cos_th, -1.0, 1.0))
        self.total_area += float(np.sum(f_norms))

        for l in range(MAX_L + 1):
            l_prefactor = np.sqrt(4.0 * np.pi / (2 * l + 1))
            for m in range(l + 1):
                ylm = _sph_harm(m, l, phi, theta)  # (N,) complex
                self._d[(l, m)] += np.sum(f_norms * l_prefactor * ylm)

    def add_facet(self, f):
        """Add a facet contribution (f = normal * area)."""
        area = np.linalg.norm(f)
        f_normalized = f / area
        cos_th = f_normalized[2]
        phi = np.arctan2(f_normalized[1], f_normalized[0])
        self.total_area += area

        for l in range(MAX_L + 1):
            l_prefactor = np.sqrt(4.0 * np.pi / (2 * l + 1))
            for m in range(l + 1):
                theta = np.arccos(np.clip(cos_th, -1, 1))
                ylm = _sph_harm(m, l, phi, theta)
                self._d[(l, m)] += area * l_prefactor * ylm

    def ql(self, l):
        """Compute q_l rotation invariant."""
        r = abs(self._d[(l, 0)])**2
        for m in range(1, l + 1):
            r += 2 * abs(self._d[(l, m)])**2
        return np.sqrt(r) / self.total_area

    def wl(self, l):
        """Compute w_l rotation invariant using Wigner 3j symbols."""
        v = 0.0 + 0.0j

        for ma in range(-l, l + 1):
            for mb in range(-l, l + 1):
                mc = -(ma + mb)
                if abs(mc) > l:
                    continue

                # Wigner 3j symbol
                w3j = _wigner3j(l, l, l, ma, mb, mc)
                if w3j == 0.0:
                    continue

                # Get coefficients, handling negative m
                a = self._get_coeff(l, ma)
                b = self._get_coeff(l, mb)
                c = self._get_coeff(l, mc)

                v += w3j * a * b * c

        if abs(v.imag) > 1e-4:
            import sys
            print(f"large spurious imaginary component (l = {l}: "
                  f"{v.real}, {v.imag}i)", file=sys.stderr)

        absv = abs(v)
        if absv == 0:
            return 0.0
        sign = 1.0 if v.real >= 0 else -1.0
        return absv**(1.0 / 3.0) / self.total_area * sign

    def _get_coeff(self, l, m):
        """Get C_lm coefficient, handling negative m via conjugate relation."""
        if m >= 0:
            return self._d[(l, m)]
        else:
            val = np.conj(self._d[(l, -m)])
            if (-m) % 2:
                val = -val
            return val


def calculate_sphmink(surface):
    """Calculate spherical Minkowski tensors for each label."""
    results = {}
    data = {}

    # Group triangles by label and batch-add facets
    unique_labels = np.unique(surface._labels)
    for lab in unique_labels:
        lab = int(lab)
        mask = surface._labels == lab
        sm = SphericalMinkowskis()
        sm.add_facets_batch(surface._normals[mask], surface._areas[mask])
        data[lab] = sm

    for label, sm in data.items():
        r = _default_sphmink()
        for l in range(MAX_L + 1):
            r.result.ql[l] = sm.ql(l)
            r.result.wl[l] = sm.wl(l)
        results[label] = r

    return results
