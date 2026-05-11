# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Keisuke Ishihara, Yajushi Khurana
# This file is part of pykarambola, a Python port of karambola.
# See LICENSE for the full license text.
"""
Tensor data structures: SymmetricMatrix3, Rank3Tensor, SymmetricRank4Tensor.
"""

import numpy as np


class SymmetricMatrix3:
    """3x3 symmetric matrix storing 6 independent elements."""

    def __init__(self):
        # Storage: st[j*(j+1)/2 + i] for i <= j
        self._st = np.zeros(6, dtype=np.float64)

    def _index(self, i, j):
        if i > j:
            i, j = j, i
        return j * (j + 1) // 2 + i

    def __getitem__(self, key):
        i, j = key
        return self._st[self._index(i, j)]

    def __setitem__(self, key, value):
        i, j = key
        self._st[self._index(i, j)] = value

    def addmul(self, prefactor, other):
        self._st += prefactor * other._st

    def __iadd__(self, other):
        self._st += other._st
        return self

    def to_numpy(self):
        m = np.zeros((3, 3), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                m[i, j] = self[i, j]
        return m

    def set_nan(self):
        self._st[:] = np.nan


class SymmetricRank4Tensor:
    """
    Symmetric rank-4 tensor stored as a 6x6 symmetric matrix (21 independent elements).
    Uses Voigt notation with sqrt(2) scaling.
    """

    def __init__(self):
        self._st = np.zeros(21, dtype=np.float64)

    def _index(self, i, j):
        if i > j:
            i, j = j, i
        return j * (j + 1) // 2 + i

    def __getitem__(self, key):
        i, j = key
        return self._st[self._index(i, j)]

    def __setitem__(self, key, value):
        i, j = key
        self._st[self._index(i, j)] = value

    def addmul(self, prefactor, other):
        self._st += prefactor * other._st

    def to_numpy(self):
        m = np.zeros((6, 6), dtype=np.float64)
        for i in range(6):
            for j in range(6):
                m[i, j] = self[i, j]
        return m

    def set_nan(self):
        self._st[:] = np.nan


class Rank3Tensor:
    """3x3x3 tensor."""

    def __init__(self):
        self._d = np.zeros((3, 3, 3), dtype=np.float64)

    def __getitem__(self, key):
        i, j, k = key
        return self._d[i, j, k]

    def __setitem__(self, key, value):
        i, j, k = key
        self._d[i, j, k] = value

    def addmul(self, prefactor, other):
        self._d += prefactor * other._d

    def to_numpy(self):
        return self._d.copy()

    def set_nan(self):
        self._d[:] = np.nan


def fourth_tensorial_power(v):
    """Compute the fourth tensorial power of a 3D vector.

    Maps a 3D vector to a SymmetricRank4Tensor using sqrt(2) scaling convention
    (Voigt notation).
    """
    sqrt2 = np.sqrt(2.0)
    x, y, z = v[0], v[1], v[2]
    t = np.array([x * x, y * y, z * z, y * z * sqrt2, x * z * sqrt2, x * y * sqrt2])
    result = SymmetricRank4Tensor()
    for i in range(6):
        for j in range(i + 1):
            result[i, j] = t[i] * t[j]
    return result
