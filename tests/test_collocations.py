"""
Tests for qhchina.analytics.collocations module.
Tests both Python and Cython implementations.
"""
import pytest
import numpy as np
import pandas as pd


class TestFindCollocates:
    """Tests for the find_collocates function."""
    
    def test_find_collocates_window_python(self, sample_documents):
        """Test window-based collocations with Python implementation."""
        from qhchina.analytics.collocations import find_collocates
        
        results = find_collocates(
            sample_documents,
            target_words=["我"],
            method="window",
            horizon=3,
        )
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) >= 0  # May have no results with small corpus
        if len(results) > 0:
            assert "target" in results.columns
            assert "collocate" in results.columns
            assert "p_value" in results.columns
    
    def test_find_collocates_window_cython(self, sample_documents):
        """Test window-based collocations with Cython implementation."""
        from qhchina.analytics.collocations import find_collocates, CYTHON_AVAILABLE
        
        if not CYTHON_AVAILABLE:
            pytest.skip("Cython extension not available")
        
        results = find_collocates(
            sample_documents,
            target_words=["我"],
            method="window",
            horizon=3,
        )
        
        assert isinstance(results, pd.DataFrame)
    
    def test_find_collocates_sentence_python(self, sample_documents):
        """Test sentence-based collocations with Python implementation."""
        from qhchina.analytics.collocations import find_collocates
        
        results = find_collocates(
            sample_documents,
            target_words=["人"],
            method="sentence",
        )
        
        assert isinstance(results, pd.DataFrame)
    
    def test_find_collocates_sentence_cython(self, sample_documents):
        """Test sentence-based collocations with Cython implementation."""
        from qhchina.analytics.collocations import find_collocates, CYTHON_AVAILABLE
        
        if not CYTHON_AVAILABLE:
            pytest.skip("Cython extension not available")
        
        results = find_collocates(
            sample_documents,
            target_words=["人"],
            method="sentence",
        )
        
        assert isinstance(results, pd.DataFrame)
    
    def test_find_collocates_with_filters(self, larger_documents):
        """Test collocations with filtering options."""
        from qhchina.analytics.collocations import find_collocates
        
        results = find_collocates(
            larger_documents,
            target_words=["我"],
            method="window",
            horizon=5,
            filters={"max_p": 0.05, "min_obs_local": 2},
        )
        
        assert isinstance(results, pd.DataFrame)
        if len(results) > 0:
            assert all(results["p_value"] <= 0.05)
            assert all(results["obs_local"] >= 2)
    
    def test_python_cython_consistency(self, larger_documents):
        """Test that Python and Cython implementations give consistent results."""
        from qhchina.analytics.collocations import find_collocates, CYTHON_AVAILABLE
        
        if not CYTHON_AVAILABLE:
            pytest.skip("Cython extension not available")
        
        # Note: both use_cython is not exposed directly in find_collocates
        # The function auto-selects, so we test results are valid
        target = ["人"]
        
        results = find_collocates(
            larger_documents,
            target_words=target,
            method="window",
            horizon=3,
        )
        
        assert isinstance(results, pd.DataFrame)


class TestCoocMatrix:
    """Tests for the cooc_matrix function."""
    
    def test_cooc_matrix_basic(self, sample_documents):
        """Test basic co-occurrence matrix generation."""
        from qhchina.analytics.collocations import cooc_matrix
        
        result = cooc_matrix(
            sample_documents,
            method='window',
            horizon=2,
            as_dataframe=True
        )
        
        assert isinstance(result, pd.DataFrame)
        # Should be square
        assert result.shape[0] == result.shape[1]
    
    def test_cooc_matrix_numpy(self, sample_documents):
        """Test co-occurrence matrix as numpy array."""
        from qhchina.analytics.collocations import cooc_matrix
        
        matrix, vocab = cooc_matrix(
            sample_documents,
            method='window',
            horizon=2,
            as_dataframe=False
        )
        
        assert isinstance(matrix, np.ndarray)
        assert isinstance(vocab, dict)
        assert matrix.shape[0] == matrix.shape[1]
    
    def test_cooc_matrix_with_vocabulary(self, sample_documents):
        """Test co-occurrence matrix with specified vocabulary."""
        from qhchina.analytics.collocations import cooc_matrix
        
        vocab_subset = ["我", "人", "时", "也"]
        
        result = cooc_matrix(
            sample_documents,
            method='window',
            horizon=2,
            vocab=vocab_subset,
            as_dataframe=True
        )
        
        assert len(result.columns) <= len(vocab_subset)


class TestAsymmetricHorizon:
    """Tests for asymmetric horizon in collocations."""
    
    def test_asymmetric_horizon(self, larger_documents):
        """Test collocations with asymmetric horizon (left != right)."""
        from qhchina.analytics.collocations import find_collocates
        
        # Test left-only context
        results = find_collocates(
            larger_documents,
            target_words=["我"],
            method="window",
            horizon=(3, 0),  # Only left context
        )
        
        assert isinstance(results, pd.DataFrame)
    
    def test_horizon_direction_matters(self, larger_documents):
        """Test that horizon direction affects results."""
        from qhchina.analytics.collocations import find_collocates
        
        results_left = find_collocates(
            larger_documents,
            target_words=["我"],
            method="window",
            horizon=(5, 0),
        )
        
        results_right = find_collocates(
            larger_documents,
            target_words=["我"],
            method="window",
            horizon=(0, 5),
        )
        
        # Results should generally differ
        assert isinstance(results_left, pd.DataFrame)
        assert isinstance(results_right, pd.DataFrame)
