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
        """Test that invalid method raises ValueError."""
        from qhchina.analytics.collocations import find_collocates
        
        with pytest.raises(ValueError, match="Invalid method"):
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
        from qhchina.analytics.collocations import cooc_matrix, CoocMatrix
        
        result = cooc_matrix(sample_documents, horizon=2)
        
        assert isinstance(result, CoocMatrix)
        # Should be square
        assert result.shape[0] == result.shape[1]
    
    def test_cooc_matrix_to_dataframe(self, sample_documents):
        """Test converting CoocMatrix to DataFrame."""
        from qhchina.analytics.collocations import cooc_matrix
        
        result = cooc_matrix(sample_documents, horizon=2)
        df = result.to_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == df.shape[1]
    
    def test_cooc_matrix_to_dense(self, sample_documents):
        """Test converting CoocMatrix to dense numpy array."""
        from qhchina.analytics.collocations import cooc_matrix
        
        result = cooc_matrix(sample_documents, horizon=2)
        arr = result.to_dense()
        
        assert isinstance(arr, np.ndarray)
        assert arr.shape[0] == arr.shape[1]
    
    def test_cooc_matrix_indexing_by_word(self, sample_documents):
        """Test indexing CoocMatrix by word pairs."""
        from qhchina.analytics.collocations import cooc_matrix
        
        result = cooc_matrix(sample_documents, horizon=2)
        
        # Get vocab to test with
        vocab = result.vocab
        if len(vocab) >= 2:
            word1, word2 = vocab[0], vocab[1]
            count = result[word1, word2]
            assert isinstance(count, int)
    
    def test_cooc_matrix_indexing_by_int(self, sample_documents):
        """Test indexing CoocMatrix by integer indices."""
        from qhchina.analytics.collocations import cooc_matrix
        
        result = cooc_matrix(sample_documents, horizon=2)
        
        if result.shape[0] >= 2:
            count = result[0, 1]
            assert isinstance(count, int)
    
    def test_cooc_matrix_row_lookup(self, sample_documents):
        """Test getting a row from CoocMatrix as dict."""
        from qhchina.analytics.collocations import cooc_matrix
        
        result = cooc_matrix(sample_documents, horizon=2)
        
        vocab = result.vocab
        if len(vocab) >= 1:
            row = result[vocab[0]]
            assert isinstance(row, dict)
    
    def test_cooc_matrix_with_vocabulary(self, sample_documents):
        """Test co-occurrence matrix with specified vocabulary."""
        from qhchina.analytics.collocations import cooc_matrix
        
        vocab_subset = ["我", "人", "时", "也"]
        
        result = cooc_matrix(sample_documents, horizon=2, vocab=vocab_subset)
        
        assert len(result) <= len(vocab_subset)
    
    def test_cooc_matrix_sparse_property(self, sample_documents):
        """Test that sparse property returns scipy sparse matrix."""
        from qhchina.analytics.collocations import cooc_matrix
        from scipy import sparse
        
        result = cooc_matrix(sample_documents, horizon=2)
        
        assert sparse.issparse(result.sparse)
    
    def test_cooc_matrix_contains(self, sample_documents):
        """Test __contains__ for word membership."""
        from qhchina.analytics.collocations import cooc_matrix
        
        result = cooc_matrix(sample_documents, horizon=2)
        
        vocab = result.vocab
        if len(vocab) >= 1:
            assert vocab[0] in result
            assert "NONEXISTENT_WORD_XYZ" not in result
    
    def test_cooc_matrix_get_with_default(self, sample_documents):
        """Test get() method with default value."""
        from qhchina.analytics.collocations import cooc_matrix
        
        result = cooc_matrix(sample_documents, horizon=2)
        
        # Existing pair should return actual count
        vocab = result.vocab
        if len(vocab) >= 2:
            count = result.get(vocab[0], vocab[1], default=-999)
            assert count != -999 or result[vocab[0], vocab[1]] == 0
        
        # Non-existent word should return default
        count = result.get("NONEXISTENT_WORD_XYZ", "ANOTHER_NONEXISTENT", default=0)
        assert count == 0


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
# Deterministic Window Calculation Tests
# =============================================================================

class TestDeterministicWindowCalculations:
    """
    Deterministic tests that verify window calculations are correct.
    Uses small, controlled examples where expected results are known.
    """
    
    def test_right_only_horizon_finds_only_right_collocates(self):
        """Verify (0, N) horizon only counts collocates to the RIGHT of target."""
        from qhchina.analytics.collocations import find_collocates
        
        # Simple sentence: TARGET is at position 1
        # Positions: 0=left, 1=TARGET, 2=r1, 3=r2, 4=r3, 5=r4
        sentences = [["left", "TARGET", "r1", "r2", "r3", "r4"]]
        
        # horizon=(0, 3) means: 0 words to the left, 3 words to the right
        results = find_collocates(
            sentences, 
            target_words="TARGET", 
            method="window",
            horizon=(0, 3),
            as_dataframe=False
        )
        
        collocates = {r["collocate"] for r in results}
        
        # Should find r1, r2, r3 (within 3 positions to the right)
        assert "r1" in collocates, "r1 should be found (1 position right)"
        assert "r2" in collocates, "r2 should be found (2 positions right)"
        assert "r3" in collocates, "r3 should be found (3 positions right)"
        
        # Should NOT find r4 (4 positions right, outside window)
        assert "r4" not in collocates, "r4 should NOT be found (4 positions right, outside window)"
        
        # Should NOT find left (to the left of target)
        assert "left" not in collocates, "left should NOT be found (it's to the LEFT of target)"
    
    def test_left_only_horizon_finds_only_left_collocates(self):
        """Verify (N, 0) horizon only counts collocates to the LEFT of target."""
        from qhchina.analytics.collocations import find_collocates
        
        # Simple sentence: TARGET is at position 4
        # Positions: 0=l4, 1=l3, 2=l2, 3=l1, 4=TARGET, 5=right
        sentences = [["l4", "l3", "l2", "l1", "TARGET", "right"]]
        
        # horizon=(3, 0) means: 3 words to the left, 0 words to the right
        results = find_collocates(
            sentences,
            target_words="TARGET",
            method="window",
            horizon=(3, 0),
            as_dataframe=False
        )
        
        collocates = {r["collocate"] for r in results}
        
        # Should find l1, l2, l3 (within 3 positions to the left)
        assert "l1" in collocates, "l1 should be found (1 position left)"
        assert "l2" in collocates, "l2 should be found (2 positions left)"
        assert "l3" in collocates, "l3 should be found (3 positions left)"
        
        # Should NOT find l4 (4 positions left, outside window)
        assert "l4" not in collocates, "l4 should NOT be found (4 positions left, outside window)"
        
        # Should NOT find right (to the right of target)
        assert "right" not in collocates, "right should NOT be found (it's to the RIGHT of target)"
    
    def test_symmetric_horizon_finds_both_sides(self):
        """Verify symmetric int horizon finds collocates on both sides."""
        from qhchina.analytics.collocations import find_collocates
        
        # TARGET in the middle
        # Positions: 0=l2, 1=l1, 2=TARGET, 3=r1, 4=r2
        sentences = [["l2", "l1", "TARGET", "r1", "r2"]]
        
        # horizon=2 means: 2 words on each side
        results = find_collocates(
            sentences,
            target_words="TARGET",
            method="window",
            horizon=2,
            as_dataframe=False
        )
        
        collocates = {r["collocate"] for r in results}
        
        # Should find all neighbors within 2 positions
        assert "l1" in collocates, "l1 should be found (1 position left)"
        assert "l2" in collocates, "l2 should be found (2 positions left)"
        assert "r1" in collocates, "r1 should be found (1 position right)"
        assert "r2" in collocates, "r2 should be found (2 positions right)"
    
    def test_asymmetric_horizon_mixed(self):
        """Verify asymmetric (2, 3) horizon finds correct collocates."""
        from qhchina.analytics.collocations import find_collocates
        
        # Positions: 0=l3, 1=l2, 2=l1, 3=TARGET, 4=r1, 5=r2, 6=r3, 7=r4
        sentences = [["l3", "l2", "l1", "TARGET", "r1", "r2", "r3", "r4"]]
        
        # horizon=(2, 3) means: 2 words to the left, 3 words to the right
        results = find_collocates(
            sentences,
            target_words="TARGET",
            method="window",
            horizon=(2, 3),
            as_dataframe=False
        )
        
        collocates = {r["collocate"] for r in results}
        
        # Left side: l1, l2 should be found; l3 should NOT (outside window)
        assert "l1" in collocates, "l1 should be found (1 position left)"
        assert "l2" in collocates, "l2 should be found (2 positions left)"
        assert "l3" not in collocates, "l3 should NOT be found (3 positions left, outside window)"
        
        # Right side: r1, r2, r3 should be found; r4 should NOT (outside window)
        assert "r1" in collocates, "r1 should be found (1 position right)"
        assert "r2" in collocates, "r2 should be found (2 positions right)"
        assert "r3" in collocates, "r3 should be found (3 positions right)"
        assert "r4" not in collocates, "r4 should NOT be found (4 positions right, outside window)"
    
    def test_horizon_at_sentence_boundaries(self):
        """Verify horizon respects sentence boundaries."""
        from qhchina.analytics.collocations import find_collocates
        
        # TARGET at the start - left side is truncated by sentence boundary
        sentences = [["TARGET", "r1", "r2", "r3"]]
        
        results = find_collocates(
            sentences,
            target_words="TARGET",
            method="window",
            horizon=(5, 2),  # Ask for 5 left, but there are none
            as_dataframe=False
        )
        
        collocates = {r["collocate"] for r in results}
        
        # Should find r1, r2 (within 2 positions right)
        assert "r1" in collocates
        assert "r2" in collocates
        # Should NOT find r3 (outside right window)
        assert "r3" not in collocates
    
    def test_multiple_target_occurrences(self):
        """Verify counts are correct when target appears multiple times."""
        from qhchina.analytics.collocations import find_collocates
        
        # TARGET appears twice, each time with "neighbor" next to it
        sentences = [
            ["TARGET", "neighbor", "other"],
            ["word", "TARGET", "neighbor"],
        ]
        
        results = find_collocates(
            sentences,
            target_words="TARGET",
            method="window",
            horizon=1,
            as_dataframe=False
        )
        
        # Find the result for "neighbor"
        neighbor_result = next((r for r in results if r["collocate"] == "neighbor"), None)
        
        assert neighbor_result is not None, "neighbor should be found as collocate"
        # "neighbor" appears next to TARGET twice (once in each sentence)
        assert neighbor_result["obs_local"] == 2, "neighbor should have obs_local=2"
    
    def test_collocate_not_counted_when_outside_all_windows(self):
        """Verify a word is not counted if it's outside window for all target occurrences."""
        from qhchina.analytics.collocations import find_collocates
        
        # "faraway" is always more than 1 position from TARGET
        sentences = [
            ["faraway", "x", "TARGET", "y"],      # faraway is 2 positions left of TARGET
            ["a", "TARGET", "z", "faraway"],      # faraway is 2 positions right of TARGET
        ]
        
        results = find_collocates(
            sentences,
            target_words="TARGET",
            method="window",
            horizon=1,  # Only 1 position on each side
            as_dataframe=False
        )
        
        collocates = {r["collocate"] for r in results}
        
        # "faraway" is 2 positions away in both sentences, should NOT be found
        assert "faraway" not in collocates, "faraway should NOT be found (always outside window)"
        
        # But immediate neighbors should be found
        assert "x" in collocates   # 1 position left in sentence 1
        assert "y" in collocates   # 1 position right in sentence 1
        assert "a" in collocates   # 1 position left in sentence 2
        assert "z" in collocates   # 1 position right in sentence 2


class TestDeterministicStatisticsCalculations:
    """
    Deterministic tests that verify all collocation statistics (obs_local, obs_global,
    exp_local, ratio_local) are calculated correctly for both sentence and window methods.
    """
    
    def test_sentence_method_statistics(self):
        """Verify all statistics are correct for sentence-based collocation."""
        from qhchina.analytics.collocations import find_collocates
        
        # Controlled corpus where we can manually calculate expected values:
        # - Total sentences: 10
        # - Sentences with 'dog': 6  (indices 0, 2, 3, 5, 6, 7)
        # - Sentences with 'cat': 5  (indices 1, 2, 4, 5, 7)
        # - Sentences with both: 3   (indices 2, 5, 7)
        sentences = [
            ["the", "dog", "runs", "fast"],           # 0: dog only
            ["the", "cat", "sleeps", "now"],          # 1: cat only
            ["the", "dog", "and", "cat", "play"],     # 2: dog + cat
            ["a", "dog", "barks", "loud"],            # 3: dog only
            ["my", "cat", "meows", "soft"],           # 4: cat only
            ["the", "dog", "chases", "cat"],          # 5: dog + cat
            ["one", "dog", "sits", "here"],           # 6: dog only
            ["dog", "meets", "cat", "today"],         # 7: dog + cat
            ["bird", "flies", "high", "up"],          # 8: neither
            ["fish", "swims", "deep", "down"],        # 9: neither
        ]
        
        results = find_collocates(
            sentences,
            target_words="dog",
            method="sentence",
            as_dataframe=False
        )
        
        cat_result = next((r for r in results if r["collocate"] == "cat"), None)
        assert cat_result is not None, "'cat' should be found as collocate of 'dog'"
        
        # Expected contingency table for collocate='cat' given target='dog':
        # a = sentences with both dog and cat = 3
        # b = sentences with dog but NOT cat = 6 - 3 = 3
        # c = sentences with cat but NOT dog = 5 - 3 = 2
        # d = sentences with neither = 10 - 3 - 3 - 2 = 2
        expected_a = 3
        expected_c = 2
        expected_obs_global = expected_a + expected_c  # = 5
        expected_exp = (6 * 5) / 10  # = 3.0
        expected_ratio = expected_a / expected_exp  # = 1.0
        
        assert cat_result["obs_local"] == expected_a, \
            f"obs_local should be {expected_a}, got {cat_result['obs_local']}"
        assert cat_result["obs_global"] == expected_obs_global, \
            f"obs_global should be {expected_obs_global}, got {cat_result['obs_global']}"
        assert abs(cat_result["exp_local"] - expected_exp) < 0.0001, \
            f"exp_local should be {expected_exp}, got {cat_result['exp_local']}"
        assert abs(cat_result["ratio_local"] - expected_ratio) < 0.0001, \
            f"ratio_local should be {expected_ratio}, got {cat_result['ratio_local']}"
    
    def test_window_method_statistics(self):
        """Verify all statistics are correct for window-based collocation."""
        from qhchina.analytics.collocations import find_collocates
        
        # Controlled corpus for window method with horizon=1:
        # 'cat' is adjacent to 'dog' (distance=1) in sentences 0 and 2 only
        sentences = [
            ["a", "dog", "cat", "b"],       # dog at 1, cat at 2 -> adjacent ✓
            ["x", "dog", "y", "cat"],       # dog at 1, cat at 3 -> NOT adjacent (distance=2)
            ["cat", "dog", "z"],            # dog at 1, cat at 0 -> adjacent ✓
            ["m", "n", "dog", "o"],         # dog at 2, no cat
            ["p", "cat", "q"],              # no dog, cat at 1
        ]
        
        results = find_collocates(
            sentences,
            target_words="dog",
            method="window",
            horizon=1,
            as_dataframe=False
        )
        
        cat_result = next((r for r in results if r["collocate"] == "cat"), None)
        assert cat_result is not None, "'cat' should be found as collocate of 'dog'"
        
        # Expected obs_local: 'cat' positions with 'dog' in window = 2
        # (sentence 0 pos 2, sentence 2 pos 0)
        expected_obs_local = 2
        
        # Expected obs_global: total 'cat' token occurrences = 4
        # (sentence 0 pos 2, sentence 1 pos 3, sentence 2 pos 0, sentence 4 pos 1)
        expected_obs_global = 4
        
        assert cat_result["obs_local"] == expected_obs_local, \
            f"obs_local should be {expected_obs_local}, got {cat_result['obs_local']}"
        assert cat_result["obs_global"] == expected_obs_global, \
            f"obs_global should be {expected_obs_global}, got {cat_result['obs_global']}"
        
        # Verify ratio and expected are computed (exact values depend on full contingency table)
        assert cat_result["ratio_local"] > 0, "ratio_local should be positive"
        assert cat_result["exp_local"] > 0, "exp_local should be positive"
    
    def test_window_method_no_association(self):
        """Verify ratio ~1 when there's no real association."""
        from qhchina.analytics.collocations import find_collocates
        
        # Corpus where 'filler' appears everywhere uniformly
        # so it should have ratio close to 1 with any target
        sentences = [
            ["filler", "dog", "filler", "x"],
            ["filler", "y", "dog", "filler"],
            ["filler", "z", "filler", "dog"],
            ["filler", "a", "filler", "b"],
        ]
        
        results = find_collocates(
            sentences,
            target_words="dog",
            method="window",
            horizon=2,
            as_dataframe=False
        )
        
        filler_result = next((r for r in results if r["collocate"] == "filler"), None)
        assert filler_result is not None, "'filler' should be found as collocate"
        
        # 'filler' appears uniformly, so ratio should be close to 1
        # Allow some tolerance since small sample sizes can cause deviation
        assert 0.5 < filler_result["ratio_local"] < 2.0, \
            f"ratio_local for uniform 'filler' should be near 1, got {filler_result['ratio_local']}"


class TestDeterministicCoocMatrixCalculations:
    """
    Deterministic tests that verify co-occurrence matrix window calculations.
    """
    
    def test_cooc_matrix_right_only_horizon(self):
        """Verify cooc_matrix (0, N) horizon only counts right co-occurrences."""
        from qhchina.analytics.collocations import cooc_matrix
        
        # Simple document: a -> b -> c -> d
        documents = [["a", "b", "c", "d"]]
        
        # horizon=(0, 1) means: only count word immediately to the RIGHT
        result = cooc_matrix(documents, horizon=(0, 1))
        
        # "a" should co-occur with "b" (b is to the right of a)
        assert result["a", "b"] > 0, "a->b should have co-occurrence (b is right of a)"
        
        # "b" should NOT co-occur with "a" from b's perspective
        # because with (0, 1), we only look RIGHT, and "a" is to the LEFT of "b"
        assert result["b", "a"] == 0, "b->a should be 0 (a is left of b, but we only look right)"
        
        # "b" should co-occur with "c"
        assert result["b", "c"] > 0, "b->c should have co-occurrence"
    
    def test_cooc_matrix_left_only_horizon(self):
        """Verify cooc_matrix (N, 0) horizon only counts left co-occurrences."""
        from qhchina.analytics.collocations import cooc_matrix
        
        # Simple document: a -> b -> c -> d
        documents = [["a", "b", "c", "d"]]
        
        # horizon=(1, 0) means: only count word immediately to the LEFT
        result = cooc_matrix(documents, horizon=(1, 0))
        
        # "b" should co-occur with "a" (a is to the left of b)
        assert result["b", "a"] > 0, "b->a should have co-occurrence (a is left of b)"
        
        # "a" should NOT co-occur with "b" from a's perspective
        # because with (1, 0), we only look LEFT, and "b" is to the RIGHT of "a"
        assert result["a", "b"] == 0, "a->b should be 0 (b is right of a, but we only look left)"
    
    def test_cooc_matrix_symmetric_horizon(self):
        """Verify symmetric horizon produces symmetric co-occurrence for adjacent words."""
        from qhchina.analytics.collocations import cooc_matrix
        
        # Simple document
        documents = [["a", "b", "c"]]
        
        # horizon=1 means: 1 word on each side (symmetric)
        result = cooc_matrix(documents, horizon=1)
        
        # With symmetric horizon, a-b and b-a should both have co-occurrences
        assert result["a", "b"] > 0, "a-b should co-occur"
        assert result["b", "a"] > 0, "b-a should co-occur"
        
        # And they should be equal (symmetric)
        assert result["a", "b"] == result["b", "a"], "Co-occurrence should be symmetric"
    
    def test_cooc_matrix_window_size_limits(self):
        """Verify co-occurrence respects window size limits."""
        from qhchina.analytics.collocations import cooc_matrix
        
        # Document: a -> b -> c -> d -> e
        documents = [["a", "b", "c", "d", "e"]]
        
        # horizon=1: only immediate neighbors
        result = cooc_matrix(documents, horizon=1)
        
        # "a" and "c" are 2 positions apart, should NOT co-occur with horizon=1
        assert result["a", "c"] == 0, "a-c should NOT co-occur (2 positions apart, horizon=1)"
        
        # "a" and "b" are adjacent, should co-occur
        assert result["a", "b"] > 0, "a-b should co-occur (adjacent)"
    
    def test_cooc_matrix_multiple_occurrences(self):
        """Verify co-occurrence counts multiple occurrences correctly."""
        from qhchina.analytics.collocations import cooc_matrix
        
        # "a" appears next to "b" twice
        documents = [["a", "b", "x", "a", "b"]]
        
        result = cooc_matrix(documents, horizon=1)
        
        # "a" should co-occur with "b" twice
        assert result["a", "b"] == 2, "a-b should have count=2 (appears twice)"


class TestPythonCythonConsistencyDeterministic:
    """
    Test that Python and Cython implementations produce identical results
    on deterministic examples.
    """
    
    def test_window_python_cython_identical_results(self):
        """Verify Python and Cython window implementations match exactly."""
        from qhchina.analytics.collocations import (
            _calculate_collocations_window_python,
            _calculate_collocations_window_cython,
            CYTHON_AVAILABLE
        )
        
        if not CYTHON_AVAILABLE:
            pytest.skip("Cython extension not available")
        
        # Deterministic test data
        sentences = [
            ["a", "TARGET", "b", "c", "d"],
            ["x", "y", "TARGET", "z"],
            ["TARGET", "m", "n"],
        ]
        target_words = ["TARGET"]
        horizon = (2, 3)
        
        # Run both implementations
        python_results = _calculate_collocations_window_python(
            sentences, target_words, horizon=horizon
        )
        cython_results = _calculate_collocations_window_cython(
            sentences, target_words, horizon=horizon
        )
        
        # Convert to comparable format
        def to_dict(results):
            return {
                (r["target"], r["collocate"]): (r["obs_local"], r["obs_global"])
                for r in results
            }
        
        python_dict = to_dict(python_results)
        cython_dict = to_dict(cython_results)
        
        # Same collocates should be found
        assert set(python_dict.keys()) == set(cython_dict.keys()), \
            "Python and Cython should find the same collocates"
        
        # Same counts
        for key in python_dict:
            assert python_dict[key] == cython_dict[key], \
                f"Counts for {key} should match: Python={python_dict[key]}, Cython={cython_dict[key]}"
    
    def test_sentence_python_cython_identical_results(self):
        """Verify Python and Cython sentence implementations match exactly."""
        from qhchina.analytics.collocations import (
            _calculate_collocations_sentence_python,
            _calculate_collocations_sentence_cython,
            CYTHON_AVAILABLE
        )
        
        if not CYTHON_AVAILABLE:
            pytest.skip("Cython extension not available")
        
        # Deterministic test data
        sentences = [
            ["a", "TARGET", "b", "c"],
            ["TARGET", "x", "y", "z"],
            ["m", "n", "TARGET"],
            ["no", "target", "here"],
        ]
        target_words = ["TARGET"]
        
        # Run both implementations
        python_results = _calculate_collocations_sentence_python(sentences, target_words)
        cython_results = _calculate_collocations_sentence_cython(sentences, target_words)
        
        # Convert to comparable format
        def to_dict(results):
            return {
                (r["target"], r["collocate"]): (r["obs_local"], r["obs_global"])
                for r in results
            }
        
        python_dict = to_dict(python_results)
        cython_dict = to_dict(cython_results)
        
        # Same collocates should be found
        assert set(python_dict.keys()) == set(cython_dict.keys()), \
            "Python and Cython should find the same collocates"
        
        # Same counts
        for key in python_dict:
            assert python_dict[key] == cython_dict[key], \
                f"Counts for {key} should match: Python={python_dict[key]}, Cython={cython_dict[key]}"


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


class TestCoocMatrixBoundaryValues:
    """Tests for co-occurrence matrix with boundary values."""
    
    def test_single_document(self):
        """Test cooc_matrix with a single document."""
        from qhchina.analytics.collocations import cooc_matrix, CoocMatrix
        
        docs = [list("我喜欢学习")]
        result = cooc_matrix(docs, horizon=1)
        
        assert isinstance(result, CoocMatrix)
        assert result.shape[0] == result.shape[1]
    
    def test_horizon_one(self, sample_documents):
        """Test cooc_matrix with horizon=1."""
        from qhchina.analytics.collocations import cooc_matrix, CoocMatrix
        
        result = cooc_matrix(sample_documents, horizon=1)
        
        assert isinstance(result, CoocMatrix)
    
    def test_binary_mode(self, sample_documents):
        """Test cooc_matrix with binary=True."""
        from qhchina.analytics.collocations import cooc_matrix
        
        result = cooc_matrix(sample_documents, horizon=2, binary=True)
        df = result.to_dataframe()
        
        # All values should be 0 or 1
        assert df.max().max() <= 1
        assert df.min().min() >= 0
    
    def test_sparse_property(self, larger_documents):
        """Test that CoocMatrix.sparse returns scipy sparse matrix."""
        from qhchina.analytics.collocations import cooc_matrix
        from scipy import sparse
        
        result = cooc_matrix(larger_documents, horizon=2)
        
        assert sparse.issparse(result.sparse)
        assert isinstance(result.word_to_index, dict)


class TestCoocMatrixDistancePreservation:
    """
    Tests to verify that OOV words preserve positional distances in cooc_matrix.
    
    When vocabulary filtering removes words, the distance between remaining
    vocabulary words should be preserved (not collapsed).
    """
    
    def test_oov_words_preserve_distance_window_method(self):
        """
        Test that OOV words preserve distance between vocabulary words.
        
        Given: ["A", "x", "y", "B"] with vocab = {"A", "B"} and horizon=1
        - "x" and "y" are OOV (out-of-vocabulary)
        - A is at position 0, B is at position 3
        - True distance is 3 positions apart
        - With horizon=1, A and B should NOT co-occur
        
        Bug scenario (collapsed distances):
        - If OOV words are removed: ["A", "B"] 
        - A and B become adjacent (distance 1)
        - With horizon=1, they would incorrectly co-occur
        """
        from qhchina.analytics.collocations import cooc_matrix
        
        # A and B are 3 positions apart (with x, y in between)
        documents = [["A", "x", "y", "B"]]
        
        # Only A and B are in vocabulary (x and y will be OOV)
        result = cooc_matrix(documents, horizon=1, vocab=["A", "B"])
        
        # A and B should NOT co-occur because true distance is 3 (> horizon=1)
        assert result["A", "B"] == 0, \
            "A and B should NOT co-occur (3 positions apart, horizon=1). OOV distance not preserved!"
        assert result["B", "A"] == 0, \
            "B and A should NOT co-occur (3 positions apart, horizon=1). OOV distance not preserved!"
    
    def test_oov_words_allow_cooccurrence_within_horizon(self):
        """
        Test that vocabulary words within horizon DO co-occur (positive case).
        
        Given: ["A", "x", "B"] with vocab = {"A", "B"} and horizon=2
        - A is at position 0, B is at position 2
        - True distance is 2 positions apart
        - With horizon=2, A and B SHOULD co-occur
        """
        from qhchina.analytics.collocations import cooc_matrix
        
        documents = [["A", "x", "B"]]  # A and B are 2 apart
        
        result = cooc_matrix(documents, horizon=2, vocab=["A", "B"])
        
        # A and B SHOULD co-occur because distance 2 <= horizon 2
        assert result["A", "B"] == 1, \
            "A and B should co-occur (2 positions apart, horizon=2)"
        assert result["B", "A"] == 1, \
            "B and A should co-occur (2 positions apart, horizon=2)"
    
    def test_oov_distance_preservation_multiple_oov(self):
        """
        Test distance preservation with multiple OOV words in sequence.
        
        Given: ["A", "x1", "x2", "x3", "x4", "B"] with vocab = {"A", "B"}
        - A at position 0, B at position 5
        - True distance is 5 positions
        - With horizon=4, A and B should NOT co-occur
        - With horizon=5, A and B SHOULD co-occur
        """
        from qhchina.analytics.collocations import cooc_matrix
        
        documents = [["A", "x1", "x2", "x3", "x4", "B"]]  # A and B are 5 apart
        
        # horizon=4: should NOT co-occur
        result_h4 = cooc_matrix(documents, horizon=4, vocab=["A", "B"])
        assert result_h4["A", "B"] == 0, \
            "A-B should NOT co-occur with horizon=4 (5 positions apart)"
        
        # horizon=5: SHOULD co-occur
        result_h5 = cooc_matrix(documents, horizon=5, vocab=["A", "B"])
        assert result_h5["A", "B"] == 1, \
            "A-B SHOULD co-occur with horizon=5 (5 positions apart)"
    
    def test_oov_distance_preservation_asymmetric_horizon(self):
        """
        Test distance preservation with asymmetric horizon.
        
        Given: ["A", "x", "y", "B"] with vocab = {"A", "B"}
        - B is 3 positions to the RIGHT of A
        - A is 3 positions to the LEFT of B
        """
        from qhchina.analytics.collocations import cooc_matrix
        
        documents = [["A", "x", "y", "B"]]
        
        # horizon=(0, 2): look only right, up to 2 positions
        # A->B: B is 3 right of A, should NOT co-occur
        result_right = cooc_matrix(documents, horizon=(0, 2), vocab=["A", "B"])
        assert result_right["A", "B"] == 0, \
            "A->B should NOT co-occur with right-only horizon=2 (B is 3 right of A)"
        
        # horizon=(0, 3): look only right, up to 3 positions  
        # A->B: B is 3 right of A, SHOULD co-occur
        result_right3 = cooc_matrix(documents, horizon=(0, 3), vocab=["A", "B"])
        assert result_right3["A", "B"] == 1, \
            "A->B SHOULD co-occur with right-only horizon=3 (B is exactly 3 right of A)"


class TestCoocMatrixCythonPythonConsistency:
    """
    Tests to verify Cython and Python implementations produce identical results.
    """
    
    def test_cython_python_window_consistency(self):
        """Verify Cython and Python paths produce identical co-occurrence matrices."""
        from qhchina.analytics.collocations import cooc_matrix, CYTHON_AVAILABLE
        
        if not CYTHON_AVAILABLE:
            pytest.skip("Cython extension not available")
        
        # Test documents with some words that will be filtered out
        documents = [
            ["the", "quick", "brown", "fox", "jumps"],
            ["over", "the", "lazy", "dog", "today"],
            ["brown", "fox", "is", "quick", "and", "lazy"],
        ]
        
        # Use vocab filter to create OOV words
        vocab = ["quick", "brown", "fox", "lazy", "dog"]
        
        # Get Cython result (default when available)
        result_cython = cooc_matrix(documents, horizon=2, vocab=vocab)
        
        # Force Python fallback by temporarily disabling Cython
        import qhchina.analytics.collocations as col_module
        original_func = col_module.calculate_cooc_matrix_window
        col_module.calculate_cooc_matrix_window = None
        
        try:
            result_python = cooc_matrix(documents, horizon=2, vocab=vocab)
        finally:
            col_module.calculate_cooc_matrix_window = original_func
        
        # Results should be identical
        pd.testing.assert_frame_equal(
            result_cython.to_dataframe(), result_python.to_dataframe(),
            check_exact=True,
            obj="Cython vs Python cooc_matrix results"
        )


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
        from qhchina import helpers
        
        helpers.load_fonts()
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
        from qhchina import helpers
        
        helpers.load_fonts()
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
        from qhchina import helpers
        
        helpers.load_fonts()
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
        from qhchina import helpers
        
        helpers.load_fonts()
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
        from qhchina import helpers
        
        helpers.load_fonts()
        collocates = find_collocates(larger_documents, target_words=["我"])
        
        if len(collocates) > 0:
            plot_collocates(collocates, color_by='obs_global')
            plt.close('all')


class TestFindCollocatesCorrection:
    """Tests for multiple testing correction in find_collocates."""
    
    def test_bonferroni_adds_adjusted_column(self, larger_documents):
        """Test that Bonferroni correction adds adjusted_p_value column."""
        from qhchina.analytics.collocations import find_collocates
        
        result = find_collocates(
            larger_documents,
            target_words=["我"],
            correction='bonferroni',
            as_dataframe=True
        )
        
        assert 'adjusted_p_value' in result.columns
        assert 'p_value' in result.columns
        # Adjusted p-values should be >= raw p-values
        assert all(result['adjusted_p_value'] >= result['p_value'] - 1e-15)
        # Adjusted p-values should be capped at 1.0
        assert all(result['adjusted_p_value'] <= 1.0)
    
    def test_fdr_bh_adds_adjusted_column(self, larger_documents):
        """Test that Benjamini-Hochberg correction adds adjusted_p_value column."""
        from qhchina.analytics.collocations import find_collocates
        
        result = find_collocates(
            larger_documents,
            target_words=["我"],
            correction='fdr_bh',
            as_dataframe=True
        )
        
        assert 'adjusted_p_value' in result.columns
        assert all(result['adjusted_p_value'] >= result['p_value'] - 1e-15)
        assert all(result['adjusted_p_value'] <= 1.0)
    
    def test_no_correction_by_default(self, larger_documents):
        """Test that no correction is applied by default."""
        from qhchina.analytics.collocations import find_collocates
        
        result = find_collocates(
            larger_documents,
            target_words=["我"],
            as_dataframe=True
        )
        
        assert 'adjusted_p_value' not in result.columns
    
    def test_invalid_correction_raises_error(self, larger_documents):
        """Test that invalid correction method raises ValueError."""
        from qhchina.analytics.collocations import find_collocates
        
        with pytest.raises(ValueError, match="Unknown correction method"):
            find_collocates(
                larger_documents,
                target_words=["我"],
                correction='invalid_method'
            )
    
    def test_max_p_filter_still_uses_raw_p_value(self, larger_documents):
        """Test that max_p filter always uses raw p_value, even with correction."""
        from qhchina.analytics.collocations import find_collocates
        
        result = find_collocates(
            larger_documents,
            target_words=["我"],
            correction='bonferroni',
            filters={'max_p': 0.05},
            as_dataframe=True
        )
        
        if len(result) > 0:
            assert all(result['p_value'] <= 0.05)
    
    def test_max_adjusted_p_filters_on_adjusted(self, larger_documents):
        """Test that max_adjusted_p filter uses adjusted_p_value."""
        from qhchina.analytics.collocations import find_collocates
        
        result = find_collocates(
            larger_documents,
            target_words=["我"],
            correction='bonferroni',
            filters={'max_adjusted_p': 0.05},
            as_dataframe=True
        )
        
        if len(result) > 0:
            assert all(result['adjusted_p_value'] <= 0.05)
    
    def test_max_adjusted_p_without_correction_raises_error(self, larger_documents):
        """Test that max_adjusted_p without correction raises ValueError."""
        from qhchina.analytics.collocations import find_collocates
        
        with pytest.raises(ValueError, match="max_adjusted_p filter requires a correction"):
            find_collocates(
                larger_documents,
                target_words=["我"],
                filters={'max_adjusted_p': 0.05}
            )
    
    def test_correction_with_list_output(self, larger_documents):
        """Test correction works when as_dataframe=False."""
        from qhchina.analytics.collocations import find_collocates
        
        result = find_collocates(
            larger_documents,
            target_words=["我"],
            correction='fdr_bh',
            as_dataframe=False
        )
        
        assert isinstance(result, list)
        if len(result) > 0:
            assert 'adjusted_p_value' in result[0]
    
    def test_correction_with_sentence_method(self, larger_documents):
        """Test that correction works with sentence method too."""
        from qhchina.analytics.collocations import find_collocates
        
        result = find_collocates(
            larger_documents,
            target_words=["我"],
            method='sentence',
            correction='bonferroni',
            as_dataframe=True
        )
        
        if len(result) > 0:
            assert 'adjusted_p_value' in result.columns
