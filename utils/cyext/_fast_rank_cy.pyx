"""Cython-accelerated Spearman rank IC calculation."""

import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt

cnp.import_array()

DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t


def rank_ic_cy(cnp.ndarray[DTYPE_t, ndim=1] factor not None,
               cnp.ndarray[DTYPE_t, ndim=1] returns not None):
    """Cython Spearman rank IC.

    Expected speedup: 5-10x over scipy.stats.spearmanr for small arrays.
    """
    cdef Py_ssize_t n = factor.shape[0]
    cdef cnp.ndarray[DTYPE_t, ndim=1] f_rank = np.empty(n, dtype=DTYPE)
    cdef cnp.ndarray[DTYPE_t, ndim=1] r_rank = np.empty(n, dtype=DTYPE)
    cdef cnp.ndarray[Py_ssize_t, ndim=1] f_order
    cdef cnp.ndarray[Py_ssize_t, ndim=1] r_order
    cdef Py_ssize_t i, valid_count
    cdef double f_mean, r_mean, cov, f_norm, r_norm
    cdef double f_val, r_val

    # Filter NaN and rank
    valid_count = 0
    for i in range(n):
        if not (np.isnan(factor[i]) or np.isnan(returns[i])):
            valid_count += 1

    if valid_count < 10:
        return np.nan

    # Use only valid values
    cdef cnp.ndarray[DTYPE_t, ndim=1] f_valid = np.empty(valid_count, dtype=DTYPE)
    cdef cnp.ndarray[DTYPE_t, ndim=1] r_valid = np.empty(valid_count, dtype=DTYPE)
    cdef Py_ssize_t idx = 0

    for i in range(n):
        if not (np.isnan(factor[i]) or np.isnan(returns[i])):
            f_valid[idx] = factor[i]
            r_valid[idx] = returns[i]
            idx += 1

    # Argsort for ranking
    f_order = np.argsort(f_valid)
    r_order = np.argsort(r_valid)

    for i in range(valid_count):
        f_rank[f_order[i]] = <double>i
        r_rank[r_order[i]] = <double>i

    # Pearson correlation of ranks
    f_mean = 0.0
    r_mean = 0.0
    for i in range(valid_count):
        f_mean += f_rank[i]
        r_mean += r_rank[i]
    f_mean /= valid_count
    r_mean /= valid_count

    cov = 0.0
    f_norm = 0.0
    r_norm = 0.0
    for i in range(valid_count):
        cov += (f_rank[i] - f_mean) * (r_rank[i] - r_mean)
        f_norm += (f_rank[i] - f_mean) * (f_rank[i] - f_mean)
        r_norm += (r_rank[i] - r_mean) * (r_rank[i] - r_mean)

    f_norm = sqrt(f_norm)
    r_norm = sqrt(r_norm)

    if f_norm < 1e-10 or r_norm < 1e-10:
        return 0.0

    return cov / (f_norm * r_norm)
