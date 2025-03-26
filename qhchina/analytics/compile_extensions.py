#!/usr/bin/env python
"""
Script to compile Cython extensions for the LDA model.
This module is part of the qhchina.analytics package.

The compile_cython_extensions() function can be called to build the optimized
extensions during package installation or runtime.
"""
import os
import sys
import subprocess
from pathlib import Path

def compile_cython_extensions(verbose=True):
    """
    Compile all Cython extensions needed for the LDA model.
    
    This function can be called during package setup or at runtime to ensure
    the optimized Cython implementations are available.
    
    Args:
        verbose: Whether to print status messages during compilation
        
    Returns:
        bool: True if compilation succeeded, False otherwise
    """
    # Get the directory of this script
    script_dir = Path(__file__).parent.absolute()
    
    # Cython extension directory
    cython_dir = script_dir / "cython_ext"
    
    # Check if the directory exists
    if not cython_dir.exists():
        if verbose:
            print(f"Error: Cython extension directory not found at {cython_dir}")
        return False
    
    # Change to the Cython directory
    original_dir = os.getcwd()
    os.chdir(cython_dir)
    
    # Run the setup.py script
    try:
        if verbose:
            print("Compiling Cython extensions...")
        
        # Capture output to hide it if not verbose
        output = None if verbose else subprocess.PIPE
        
        # Run the compilation
        subprocess.check_call(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            stdout=output,
            stderr=output
        )
        
        if verbose:
            print("Successfully compiled Cython extensions!")
        return True
    except subprocess.CalledProcessError as e:
        if verbose:
            print(f"Error compiling Cython extensions: {e}")
        return False
    except Exception as e:
        if verbose:
            print(f"Unexpected error during Cython compilation: {e}")
        return False
    finally:
        # Change back to the original directory
        os.chdir(original_dir)

# Expose the compilation function in the module
__all__ = ['compile_cython_extensions']

if __name__ == "__main__":
    if compile_cython_extensions():
        print("You can now import the LDA model with Cython optimizations.")
    else:
        print("Failed to compile Cython extensions. LDA model will use pure Python implementation.") 