"""
Setup script for compiling Cython extensions for LDA.
"""
from setuptools import setup, Extension
import sys
import platform
import os

try:
    from Cython.Build import cythonize
    import numpy as np
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install the required packages: pip install cython numpy")
    sys.exit(1)

# Parse extensions to compile
extensions_to_compile = None
for arg in sys.argv:
    if arg.startswith("--extensions="):
        # Parse the extensions argument and remove it from sys.argv
        _, extensions_str = arg.split("=", 1)
        extensions_to_compile = [ext.strip() for ext in extensions_str.split(",") if ext.strip()]
        sys.argv.remove(arg)
        break

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

# Get the directory of this script to determine relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

extensions = []

# Define all available extensions
available_extensions = {
    "lda_sampler": Extension(
        "lda_sampler",
        sources=[os.path.join(SCRIPT_DIR, "lda_sampler.pyx")],
        include_dirs=[np.get_include()],
        language="c",
        extra_compile_args=extra_compile_args,
    ),
    "word2vec": Extension(
        "word2vec",
        sources=[os.path.join(SCRIPT_DIR, "word2vec.pyx")],
        include_dirs=[np.get_include()],
        language="c",
        extra_compile_args=extra_compile_args,
    )
}

# Add extensions based on filter
if extensions_to_compile:
    # Add only the specified extensions that exist
    for ext_name in extensions_to_compile:
        if ext_name in available_extensions:
            extensions.append(available_extensions[ext_name])
        else:
            print(f"Warning: Extension '{ext_name}' is not defined in setup.py")
else:
    # Add all extensions if no filter is provided
    extensions = list(available_extensions.values())

setup(
    name="cython_extensions",
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
    include_dirs=[np.get_include()]
) 