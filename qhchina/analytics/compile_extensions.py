#!/usr/bin/env python
"""
Utility script to compile all Cython extensions.

This script handles the compilation process for all Cython extensions in the project.
Run this script to build the extensions in place.

Options:
    --use-native: Enable architecture-specific optimizations (adds -march=native)
    --debug: Build in debug mode with fewer optimizations
    --clean: Clean build directories before compiling
    --extensions=ext1,ext2,...: Specify which extensions to compile (comma-separated, no spaces)
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path

# Get the current directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CYTHON_DIR = os.path.join(SCRIPT_DIR, "cython_ext")

# Available extensions in the project
AVAILABLE_EXTENSIONS = ["lda_sampler", "word2vec"]

def clean_build_files():
    """Clean up build artifacts."""
    print("Cleaning build files...")
    # Clean intermediate files
    extensions = [".c", ".cpp", ".so", ".pyd", ".html"]
    
    # Find and remove all build directories
    for root, dirs, files in os.walk(CYTHON_DIR):
        # Remove build directories
        if "build" in dirs:
            build_dir = os.path.join(root, "build")
            print(f"Removing {build_dir}")
            shutil.rmtree(build_dir, ignore_errors=True)
        
        # Remove compiled files
        for file in files:
            name, ext = os.path.splitext(file)
            if ext in extensions or file.endswith(".cpython-*.so"):
                file_path = os.path.join(root, file)
                print(f"Removing {file_path}")
                try:
                    os.remove(file_path)
                except OSError as e:
                    print(f"Error removing {file_path}: {e}")

def check_available_extensions():
    """Check and print which extensions are available."""
    compiled_extensions = []
    for ext_file in Path(CYTHON_DIR).glob("*.so"):
        compiled_extensions.append(ext_file.stem)
    for ext_file in Path(CYTHON_DIR).glob("*.pyd"):  # Windows
        compiled_extensions.append(ext_file.stem)
    
    for ext in AVAILABLE_EXTENSIONS:
        if ext in compiled_extensions:
            print(f"  - {ext.capitalize()}: AVAILABLE")
        else:
            print(f"  - {ext.capitalize()}: NOT COMPILED")
    
    print("\nNote: If you're missing an extension, check the build log for errors.")

def compile_extensions(extensions_to_compile=None):
    """
    Compile Cython extensions.
    
    Args:
        extensions_to_compile (list, optional): List of extensions to compile.
            If None, will check for --extensions CLI argument. 
            If neither is provided, compiles all available extensions.
    """
    if extensions_to_compile:
        print(f"Compiling specific extensions: {', '.join(extensions_to_compile)}")
    # If no extensions provided as argument, extract from CLI args if available
    if extensions_to_compile is None:
        # Extract specific extensions if provided via command line
        extensions_arg = next((arg for arg in sys.argv if arg.startswith("--extensions=")), None)
        
        if extensions_arg:
            # Parse the extensions to compile
            _, extensions_str = extensions_arg.split("=", 1)
            extensions_to_compile = [ext.strip() for ext in extensions_str.split(",") if ext.strip()]
    
    # If we have extensions to compile (either from args or CLI), validate them
    if extensions_to_compile:
        # Filter out any invalid extension names
        valid_extensions = [ext for ext in extensions_to_compile if ext in AVAILABLE_EXTENSIONS]
        invalid_extensions = [ext for ext in extensions_to_compile if ext not in AVAILABLE_EXTENSIONS]
        
        if invalid_extensions:
            print(f"Warning: Unknown extensions specified: {', '.join(invalid_extensions)}")
            print(f"Available extensions are: {', '.join(AVAILABLE_EXTENSIONS)}")
        
        if not valid_extensions:
            print("No valid extensions to compile.")
            return
        
        extensions_to_compile = valid_extensions
        print(f"Compiling specific extensions: {', '.join(extensions_to_compile)}")
    else:
        # If no extensions specified, compile all available ones
        extensions_to_compile = AVAILABLE_EXTENSIONS
        print("Compiling all available extensions")
    
    # Change to the Cython extensions directory
    os.chdir(CYTHON_DIR)
    
    # Prepare command with any additional arguments
    cmd = [sys.executable, "setup.py", "build_ext", "--inplace"]
    
    # Add extensions to compile if specific ones were requested
    if extensions_to_compile:
        cmd.append(f"--extensions={','.join(extensions_to_compile)}")
    
    # Add any additional arguments
    for arg in sys.argv[1:]:
        if arg == "--clean" or arg.startswith("--extensions="):
            continue
        cmd.append(arg)
    
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print("\nCompilation successful! The following extensions are available:")
        check_available_extensions()
    except subprocess.CalledProcessError as e:
        print(f"Compilation failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if "--clean" in sys.argv:
        clean_build_files()
    
    compile_extensions() 