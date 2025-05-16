#!/usr/bin/env python
"""
Setup script for qhchina package.
This file is used for building distribution packages.
"""
from setuptools import setup, Extension, find_packages
import os
import sys
import platform
import numpy

try:
    from Cython.Build import cythonize
except ImportError:
    def cythonize(extensions, **kwargs):
        # If Cython is not available, fallback to C sources
        for extension in extensions:
            sources = []
            for source in extension.sources:
                path, ext = os.path.splitext(source)
                if ext == '.pyx':
                    # Use relative path for the C file
                    rel_path = os.path.relpath(path + '.c', os.path.dirname(__file__))
                    if os.path.exists(rel_path):
                        sources.append(rel_path)
                    else:
                        raise ValueError(f"Cython source {source} not found and no pre-generated C file exists")
                else:
                    sources.append(source)
            extension.sources = sources
        return extensions

# Determine platform-specific compiler arguments
extra_compile_args = []
if platform.system() == "Windows":
    extra_compile_args = ["/O2"]  # Optimization for Windows
else:
    # Unix-like systems (Linux, macOS)
    extra_compile_args = ["-O3"]

# Use relative paths for all sources
extensions = [
    Extension(
        "qhchina.analytics.cython_ext.lda_sampler",
        sources=["qhchina/analytics/cython_ext/lda_sampler.pyx"],
        include_dirs=[numpy.get_include()],
        language="c",
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "qhchina.analytics.cython_ext.word2vec",
        sources=["qhchina/analytics/cython_ext/word2vec.pyx"],
        include_dirs=[numpy.get_include()],
        language="c",
        extra_compile_args=extra_compile_args,
    )
]

# Main setup configuration
setup(
    packages=find_packages(),
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "cdivision": True,
        },
    ),
    include_dirs=[numpy.get_include()]
)