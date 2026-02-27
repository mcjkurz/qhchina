#!/usr/bin/env python
"""
Setup script for qhchina package.
This file is used for building distribution packages.
"""
from setuptools import setup, Extension, find_packages
import os
import sys
import platform
import tempfile
import shutil
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


def _check_openmp():
    """Try to compile a small program with OpenMP; return (compile_flags, link_flags) or None."""
    from distutils.ccompiler import new_compiler
    from distutils.errors import CompileError, LinkError

    if platform.system() == "Windows":
        candidates = [(["/openmp"], [])]
    elif platform.system() == "Darwin":
        candidates = [
            (["-Xpreprocessor", "-fopenmp"], ["-lomp"]),
            (["-fopenmp"], ["-fopenmp"]),
        ]
    else:
        candidates = [(["-fopenmp"], ["-fopenmp"])]

    tmpdir = tempfile.mkdtemp()
    try:
        src = os.path.join(tmpdir, "omp_test.c")
        with open(src, "w") as f:
            f.write('#include <omp.h>\nint main(void) { return omp_get_num_threads(); }\n')

        compiler = new_compiler()
        compiler.add_include_dir(numpy.get_include())

        for comp_flags, link_flags in candidates:
            try:
                objs = compiler.compile([src], output_dir=tmpdir, extra_postargs=comp_flags)
                compiler.link_executable(objs, os.path.join(tmpdir, "omp_test"),
                                         extra_postargs=link_flags)
                return comp_flags, link_flags
            except (CompileError, LinkError):
                continue
    except Exception:
        pass
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return None


# Determine platform-specific compiler arguments
extra_compile_args = []
if platform.system() == "Windows":
    extra_compile_args = ["/O2"]
else:
    extra_compile_args = ["-O3"]

# Detect OpenMP (used only for fisher extension)
_omp = _check_openmp()
if _omp is not None:
    _omp_compile, _omp_link = _omp
    fisher_compile_args = extra_compile_args + _omp_compile
    fisher_link_args = _omp_link
    print("setup.py: OpenMP detected — fisher extension will use parallel loops")
else:
    fisher_compile_args = extra_compile_args
    fisher_link_args = []
    print("setup.py: OpenMP not available — fisher extension will use single-threaded loops")

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
    ),
    Extension(
        "qhchina.analytics.cython_ext.collocations",
        sources=["qhchina/analytics/cython_ext/collocations.pyx"],
        include_dirs=[numpy.get_include()],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "qhchina.analytics.cython_ext.fisher",
        sources=["qhchina/analytics/cython_ext/fisher.pyx"],
        include_dirs=[numpy.get_include()],
        language="c",
        extra_compile_args=fisher_compile_args,
        extra_link_args=fisher_link_args,
    )
]

# Main setup configuration
setup(
    packages=find_packages(exclude=[
        "tests", "tests.*",
        "test", "test.*",
        "mytest", "mytest.*",
        "scripts", "scripts.*",
        "venv", "venv.*",
        "build", "build.*",
        "dist", "dist.*",
    ]),
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