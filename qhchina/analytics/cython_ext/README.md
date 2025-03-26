# LDA Cython Extensions

This directory contains Cython optimizations for the LDA Gibbs sampler. The main goal is to accelerate the sampling process, which is the main bottleneck in LDA implementations.

## Installation

To compile the Cython extensions, you need to have the following dependencies:

- Cython
- A C compiler (GCC, Clang, MSVC, etc.)
- NumPy

You can install these with:

```bash
pip install cython numpy
```

Then compile the extensions by running:

```bash
python ../compile_extensions.py
```

Or from the main directory:

```bash
python -m analytics.compile_extensions
```

## Performance Improvements

The Cython implementation offers several performance improvements:

1. **Static Typing**: All key variables use C types instead of Python objects
2. **Memory Views**: Efficient access to NumPy arrays
3. **Optimized Multinomial Sampling**: Custom sampling implementation
4. **Compiler Optimizations**: Using -O3, -ffast-math, etc.

In our testing, the Cython implementation is approximately 10-50x faster than the pure Python version, depending on the dataset size and number of topics.

## Implementation Details

The key optimizations in the Cython code:

- `sample_topic()`: The core sampling function that calculates the topic probability distribution and samples a new topic
- `run_iteration()`: An optimized function that runs a full Gibbs sampling iteration over all documents and words
- `_sample_multinomial()`: An efficient function to sample from a discrete probability distribution

## Fallback Mechanism

The main LDA implementation in `topicmodels.py` will try to use the Cython implementation first, but will fall back to the pure Python implementation if the Cython extensions are not available or could not be compiled. 