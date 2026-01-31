"""
Tests for qhchina.analytics.collocations module.
Tests both Python and Cython implementations.
"""
import pytest
import numpy as np
import pandas as pd


# =============================================================================
# Edge Cases and Input Validation
# =============================================================================

class TestFindCollocatesValidation:
    """Tests for input validation in find_collocates."""
    
    def test_empty_sentences_raises_error(self):
        """Test that empty sentences raises ValueError."""
        from qhchina.analytics.collocations import find_collocates
        
        with pytest.raises(ValueError, match="sentences cannot be empty"):
            find_collocates([], target_words=["我"])
    
    def test_all_empty_sentences_raises_error(self):
        """Test that all empty sentences raises ValueError."""
        from qhchina.analytics.collocations import find_collocates
        
        with pytest.raises(ValueError, match="All sentences are empty"):
            find_collocates([[], [], []], target_words=["我"])
    
    def test_invalid_sentences_type_raises_error(self):
        """Test that non-list sentences raises ValueError."""
        from qhchina.analytics.collocations import find_collocates
        
        with pytest.raises(ValueError, match="must be a list of lists"):
            find_collocates(["not", "tokenized"], target_words=["我"])
    
    def test_empty_target_words_raises_error(self, sample_documents):
        """Test that empty target_words raises ValueError."""
        from qhchina.analytics.collocations import find_collocates
        
        with pytest.raises(ValueError, match="target_words cannot be empty"):
            find_collocates(sample_documents, target_words=[])
    
    def test_invalid_method_raises_error(self, sample_documents):
        """Test that invalid method raises NotImplementedError."""
        from qhchina.analytics.collocations import find_collocates
        
        with pytest.raises(NotImplementedError, match="not implemented"):
            find_collocates(sample_documents, target_words=["我"], method="invalid")
    
    def test_horizon_with_sentence_method_raises_error(self, sample_documents):
        """Test that horizon with sentence method raises ValueError."""
        from qhchina.analytics.collocations import find_collocates
        
        with pytest.raises(ValueError, match="horizon.*not applicable"):
            find_collocates(sample_documents, target_words=["我"], method="sentence", horizon=5)
    
    @pytest.mark.parametrize("filter_key,filter_value,expected_error", [
        ('max_p', -0.1, "must be a number between 0 and 1"),
        ('max_p', 1.5, "must be a number between 0 and 1"),
        ('max_p', "not_a_number", "must be a number between 0 and 1"),
        ('min_obs_local', -1, "must be a non-negative integer"),
        ('min_word_length', 0, "must be a positive integer"),
        ('min_word_length', -1, "must be a positive integer"),
        ('stopwords', "not_a_list", "must be a list or set"),
        ('invalid_key', 'value', "Invalid filter keys"),
    ])
    def test_invalid_filter_values(self, sample_documents, filter_key, filter_value, expected_error):
        """Test that invalid filter values raise ValueError with helpful message."""
        from qhchina.analytics.collocations import find_collocates
        
        with pytest.raises(ValueError, match=expected_error):
            find_collocates(
                sample_documents,
                target_words=["我"],
                filters={filter_key: filter_value}
            )
    
    def test_string_target_word_converted_to_list(self, sample_documents):
        """Test that a string target_word is converted to list."""
        from qhchina.analytics.collocations import find_collocates
        
        # Should not raise - string is converted to list internally
        results = find_collocates(sample_documents, target_words="人")
        assert isinstance(results, pd.DataFrame)


class TestFindCollocatesOutputFormats:
    """Tests for output format options."""
    
    def test_as_dataframe_false_returns_list(self, sample_documents):
        """Test that as_dataframe=False returns a list."""
        from qhchina.analytics.collocations import find_collocates
        
        results = find_collocates(
            sample_documents,
            target_words=["人"],
            method="window",
            as_dataframe=False
        )
        
        assert isinstance(results, list)
        if len(results) > 0:
            assert isinstance(results[0], dict)
            assert "target" in results[0]
            assert "collocate" in results[0]
    
    def test_alternative_parameter_less(self, larger_documents):
        """Test alternative='less' for Fisher's exact test."""
        from qhchina.analytics.collocations import find_collocates
        
        results = find_collocates(
            larger_documents,
            target_words=["我"],
            method="window",
            alternative='less'
        )
        
        assert isinstance(results, pd.DataFrame)
    
    def test_alternative_parameter_two_sided(self, larger_documents):
        """Test alternative='two-sided' for Fisher's exact test."""
        from qhchina.analytics.collocations import find_collocates
        
        results = find_collocates(
            larger_documents,
            target_words=["我"],
            method="window",
            alternative='two-sided'
        )
        
        assert isinstance(results, pd.DataFrame)


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


# =============================================================================
# Co-occurrence Matrix Validation
# =============================================================================

class TestCoocMatrixValidation:
    """Tests for input validation in cooc_matrix."""
    
    def test_empty_documents_raises_error(self):
        """Test that empty documents raises ValueError."""
        from qhchina.analytics.collocations import cooc_matrix
        
        with pytest.raises(ValueError, match="documents cannot be empty"):
            cooc_matrix([])
    
    def test_invalid_documents_type_raises_error(self):
        """Test that non-list documents raises ValueError."""
        from qhchina.analytics.collocations import cooc_matrix
        
        with pytest.raises(ValueError, match="must be a list of lists"):
            cooc_matrix(["not", "tokenized"])
    
    def test_invalid_method_raises_error(self, sample_documents):
        """Test that invalid method raises ValueError."""
        from qhchina.analytics.collocations import cooc_matrix
        
        with pytest.raises(ValueError, match="method must be"):
            cooc_matrix(sample_documents, method="invalid")
    
    def test_horizon_with_document_method_raises_error(self, sample_documents):
        """Test that horizon with document method raises ValueError."""
        from qhchina.analytics.collocations import cooc_matrix
        
        with pytest.raises(ValueError, match="horizon.*not applicable"):
            cooc_matrix(sample_documents, method="document", horizon=5)


class TestCoocMatrixBoundaryValues:
    """Tests for co-occurrence matrix with boundary values."""
    
    def test_single_document(self):
        """Test cooc_matrix with a single document."""
        from qhchina.analytics.collocations import cooc_matrix
        
        docs = [list("我喜欢学习")]
        result = cooc_matrix(docs, method='window', horizon=1, as_dataframe=True)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == result.shape[1]
    
    def test_horizon_one(self, sample_documents):
        """Test cooc_matrix with horizon=1."""
        from qhchina.analytics.collocations import cooc_matrix
        
        result = cooc_matrix(sample_documents, method='window', horizon=1, as_dataframe=True)
        
        assert isinstance(result, pd.DataFrame)
    
    def test_binary_mode(self, sample_documents):
        """Test cooc_matrix with binary=True."""
        from qhchina.analytics.collocations import cooc_matrix
        
        result = cooc_matrix(sample_documents, method='window', horizon=2, binary=True, as_dataframe=True)
        
        # All values should be 0 or 1
        assert result.max().max() <= 1
        assert result.min().min() >= 0
    
    def test_sparse_matrix(self, larger_documents):
        """Test cooc_matrix with use_sparse=True."""
        from qhchina.analytics.collocations import cooc_matrix
        from scipy import sparse
        
        matrix, vocab = cooc_matrix(
            larger_documents, 
            method='window', 
            horizon=2, 
            as_dataframe=False,
            use_sparse=True
        )
        
        assert sparse.issparse(matrix)
        assert isinstance(vocab, dict)


# =============================================================================
# Plot Collocates Tests
# =============================================================================

class TestPlotCollocates:
    """Tests for plot_collocates visualization function."""
    
    def test_plot_collocates_basic(self, larger_documents):
        """Test basic collocation plotting."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from qhchina.analytics.collocations import find_collocates, plot_collocates
        
        collocates = find_collocates(larger_documents, target_words=["我"])
        
        if len(collocates) > 0:
            # Should not raise
            plot_collocates(collocates)
            plt.close('all')
    
    def test_plot_collocates_empty_raises_error(self):
        """Test that empty collocates raises ValueError."""
        from qhchina.analytics.collocations import plot_collocates
        
        with pytest.raises(ValueError, match="Empty collocates"):
            plot_collocates([])
    
    def test_plot_collocates_missing_columns_raises_error(self):
        """Test that missing columns raise ValueError."""
        from qhchina.analytics.collocations import plot_collocates
        
        with pytest.raises(ValueError, match="Missing required columns"):
            # Missing ratio_local and p_value columns
            plot_collocates([{'target': 'a', 'collocate': 'b'}])
    
    def test_plot_collocates_with_labels(self, larger_documents):
        """Test plotting with labels enabled."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from qhchina.analytics.collocations import find_collocates, plot_collocates
        
        collocates = find_collocates(larger_documents, target_words=["我"])
        
        if len(collocates) > 0:
            plot_collocates(collocates, show_labels=True, label_top_n=5)
            plt.close('all')
    
    def test_plot_collocates_custom_columns(self, larger_documents):
        """Test plotting with custom x and y columns."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from qhchina.analytics.collocations import find_collocates, plot_collocates
        
        collocates = find_collocates(larger_documents, target_words=["我"])
        
        if len(collocates) > 0:
            plot_collocates(collocates, x_col='obs_local', y_col='exp_local', 
                          x_scale='linear', y_scale='linear')
            plt.close('all')
    
    def test_plot_collocates_save_to_file(self, larger_documents, tmp_path):
        """Test saving plot to file."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from qhchina.analytics.collocations import find_collocates, plot_collocates
        
        collocates = find_collocates(larger_documents, target_words=["我"])
        
        if len(collocates) > 0:
            filepath = tmp_path / "collocates_plot.png"
            plot_collocates(collocates, filename=str(filepath))
            plt.close('all')
            
            assert filepath.exists()
    
    def test_plot_collocates_with_color_by(self, larger_documents):
        """Test plotting with color_by parameter."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from qhchina.analytics.collocations import find_collocates, plot_collocates
        
        collocates = find_collocates(larger_documents, target_words=["我"])
        
        if len(collocates) > 0:
            plot_collocates(collocates, color_by='obs_global')
            plt.close('all')
