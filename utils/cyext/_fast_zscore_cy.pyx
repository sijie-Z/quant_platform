"""Cython-accelerated cross-sectional z-score normalization."""

import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, isnan

cnp.import_array()

DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t


def zscore_cross_section_cy(cnp.ndarray[DTYPE_t, ndim=1] values not None):
    """Cython cross-sectional z-score.

    Expected speedup: 3-8x over Pandas.
    """
    cdef Py_ssize_t n = values.shape[0]
    cdef cnp.ndarray[DTYPE_t, ndim=1] result = np.empty(n, dtype=DTYPE)
    cdef Py_ssize_t i, count
    cdef double mean, m2, val, delta, std

    # Single-pass mean and variance (Welford)
    count = 0
    mean = 0.0
    m2 = 0.0

    for i in range(n):
        val = values[i]
        if isnan(val):
            result[i] = np.nan
            continue
        result[i] = 0.0  # placeholder
        count += 1
        delta = val - mean
        mean += delta / count
        m2 += delta * (val - mean)

    if count < 2:
        return result

    std = sqrt(m2 / (count - 1))
    if std < 1e-10:
        return result

    # Normalize
    for i in range(n):
        if not isnan(values[i]):
            result[i] = (values[i] - mean) / std

    return result
