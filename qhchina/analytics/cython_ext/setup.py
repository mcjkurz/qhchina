"""
Setup script for compiling Cython extensions for LDA.
"""
from setuptools import setup, Extension
import sys
import platform

try:
    from Cython.Build import cythonize
    import numpy as np
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install the required packages: pip install cython numpy")
    sys.exit(1)

# Determine platform-specific compiler arguments
extra_compile_args = []
if platform.system() == "Windows":
    extra_compile_args = ["/O2"]  # Optimization for Windows
else:
    # Unix-like systems (Linux, macOS)
    extra_compile_args = ["-O3"]
    
    # Additional optimizations for non-Windows platforms
    if platform.system() != "Darwin" or not platform.machine().startswith('arm'):
        # Fast math can cause issues on Apple Silicon
        extra_compile_args.append("-ffast-math")
    
    # Native architecture optimizations - can cause compatibility issues if distributing binaries
    # Only use if building for local use
    if "--use-native" in sys.argv:
        extra_compile_args.append("-march=native")
        sys.argv.remove("--use-native")

extensions = [
    Extension(
        "lda_sampler",
        ["lda_sampler.pyx"],
        include_dirs=[np.get_include()],
        language="c",
        extra_compile_args=extra_compile_args
    )
]

setup(
    name="lda_cython_extensions",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': 3,
            'boundscheck': False,
            'wraparound': False,
            'initializedcheck': False,
        }
    ),
    include_dirs=[np.get_include()]
) 