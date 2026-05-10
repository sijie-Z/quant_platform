"""Setup script for compiling Cython extensions.

Usage:
    # Compile all extensions
    python setup.py build_ext --inplace

    # Compile with OpenMP parallelization (Linux/Mac)
    python setup.py build_ext --inplace --parallel

    # Windows: need Visual Studio Build Tools
    # Install: pip install setuptools Cython numpy
"""

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        "quant_platform.utils.cyext._fast_rolling_cy",
        sources=["quant_platform/utils/cyext/_fast_rolling_cy.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["/O2"] if __import__("sys").platform == "win32" else ["-O3", "-march=native"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "quant_platform.utils.cyext._fast_rank_cy",
        sources=["quant_platform/utils/cyext/_fast_rank_cy.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["/O2"] if __import__("sys").platform == "win32" else ["-O3", "-march=native"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "quant_platform.utils.cyext._fast_zscore_cy",
        sources=["quant_platform/utils/cyext/_fast_zscore_cy.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["/O2"] if __import__("sys").platform == "win32" else ["-O3", "-march=native"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
]

setup(
    name="quant_platform_cyext",
    ext_modules=cythonize(extensions, compiler_directives={
        "boundscheck": False,
        "wraparound": False,
        "cdivision": True,
        "language_level": "3",
    }),
)
