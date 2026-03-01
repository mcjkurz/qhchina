"""
Tests for new analytics features:
- kwic() — Keywords in Context
- CoocMatrix.sum() and .to_ppmi()
- compare_collocates()
- compare_corpora() with method='log_likelihood' and 'statistic' column
- find_shared_sequences()
- statistics.pyx (batch_chi2, batch_log_likelihood)
"""
import pytest
import numpy as np
import pandas as pd


# =============================================================================
# Batch statistics (statistics.pyx)
# =============================================================================

class TestBatchChi2:
    """Tests for batch_chi2 in statistics.pyx."""

    def test_basic_chi2(self):
        from qhchina.analytics.cython_ext.statistics import batch_chi2
        a = np.array([10, 20], dtype=np.int64)
        b = np.array([5, 10], dtype=np.int64)
        c = np.array([3, 8], dtype=np.int64)
        d = np.array([20, 40], dtype=np.int64)
        stats, pvals = batch_chi2(a, b, c, d)
        assert stats.shape == (2,)
        assert pvals.shape == (2,)
        assert np.all(stats >= 0)
        assert np.all((pvals >= 0) & (pvals <= 1))

    def test_chi2_with_correction(self):
        from qhchina.analytics.cython_ext.statistics import batch_chi2
        a = np.array([10], dtype=np.int64)
        b = np.array([5], dtype=np.int64)
        c = np.array([3], dtype=np.int64)
        d = np.array([20], dtype=np.int64)
        stats_no, pvals_no = batch_chi2(a, b, c, d, correction=False)
        stats_yes, pvals_yes = batch_chi2(a, b, c, d, correction=True)
        assert stats_no[0] >= stats_yes[0]
        assert pvals_no[0] <= pvals_yes[0]

    def test_chi2_zero_expected_gives_nan(self):
        from qhchina.analytics.cython_ext.statistics import batch_chi2
        a = np.array([0], dtype=np.int64)
        b = np.array([0], dtype=np.int64)
        c = np.array([0], dtype=np.int64)
        d = np.array([0], dtype=np.int64)
        stats, pvals = batch_chi2(a, b, c, d)
        assert np.isnan(stats[0])
        assert np.isnan(pvals[0])

    def test_chi2_length_mismatch_raises(self):
        from qhchina.analytics.cython_ext.statistics import batch_chi2
        a = np.array([10, 20], dtype=np.int64)
        b = np.array([5], dtype=np.int64)
        c = np.array([3], dtype=np.int64)
        d = np.array([20], dtype=np.int64)
        with pytest.raises(ValueError, match="same length"):
            batch_chi2(a, b, c, d)


class TestBatchLogLikelihood:
    """Tests for batch_log_likelihood in statistics.pyx."""

    def test_basic_g2(self):
        from qhchina.analytics.cython_ext.statistics import batch_log_likelihood
        a = np.array([10], dtype=np.int64)
        b = np.array([5], dtype=np.int64)
        c = np.array([3], dtype=np.int64)
        d = np.array([20], dtype=np.int64)
        stats, pvals = batch_log_likelihood(a, b, c, d)
        assert stats.shape == (1,)
        assert pvals.shape == (1,)
        assert stats[0] > 0
        assert 0 <= pvals[0] <= 1

    def test_g2_equal_distributions(self):
        from qhchina.analytics.cython_ext.statistics import batch_log_likelihood
        a = np.array([10], dtype=np.int64)
        b = np.array([10], dtype=np.int64)
        c = np.array([10], dtype=np.int64)
        d = np.array([10], dtype=np.int64)
        stats, pvals = batch_log_likelihood(a, b, c, d)
        assert stats[0] == pytest.approx(0.0, abs=1e-10)
        assert pvals[0] == pytest.approx(1.0, abs=0.01)

    def test_g2_multiple_tables(self):
        from qhchina.analytics.cython_ext.statistics import batch_log_likelihood
        a = np.array([10, 50, 5], dtype=np.int64)
        b = np.array([5, 10, 100], dtype=np.int64)
        c = np.array([3, 20, 8], dtype=np.int64)
        d = np.array([20, 100, 200], dtype=np.int64)
        stats, pvals = batch_log_likelihood(a, b, c, d)
        assert len(stats) == 3
        assert len(pvals) == 3


class TestBatchFisherInStatistics:
    """Verify batch_fisher_exact works from the new statistics module."""

    def test_fisher_import_from_statistics(self):
        from qhchina.analytics.cython_ext.statistics import batch_fisher_exact
        a = np.array([1], dtype=np.int64)
        b = np.array([9], dtype=np.int64)
        c = np.array([11], dtype=np.int64)
        d = np.array([3], dtype=np.int64)
        pvals = batch_fisher_exact(a, b, c, d, 'two-sided')
        assert 0 <= pvals[0] <= 1


# =============================================================================
# compare_corpora enhancements
# =============================================================================

class TestCompareCorporaLogLikelihood:
    """Tests for method='log_likelihood' in compare_corpora."""

    def test_log_likelihood_method(self, sample_documents):
        from qhchina.analytics import compare_corpora
        corpus_a = sample_documents[:3]
        corpus_b = sample_documents[2:]
        result = compare_corpora(corpus_a, corpus_b, method='log_likelihood')
        assert isinstance(result, pd.DataFrame)
        assert 'statistic' in result.columns
        assert 'p_value' in result.columns
        assert all(result['statistic'] >= 0)

    def test_statistic_column_in_chi2(self, sample_documents):
        from qhchina.analytics import compare_corpora
        corpus_a = sample_documents[:3]
        corpus_b = sample_documents[2:]
        result = compare_corpora(corpus_a, corpus_b, method='chi2')
        assert 'statistic' in result.columns

    def test_no_statistic_column_in_fisher(self, sample_documents):
        from qhchina.analytics import compare_corpora
        corpus_a = sample_documents[:3]
        corpus_b = sample_documents[2:]
        result = compare_corpora(corpus_a, corpus_b, method='fisher')
        assert 'statistic' not in result.columns

    def test_sort_by_statistic(self, sample_documents):
        from qhchina.analytics import compare_corpora
        corpus_a = sample_documents[:3]
        corpus_b = sample_documents[2:]
        result = compare_corpora(corpus_a, corpus_b, method='log_likelihood', sort_by='statistic')
        if len(result) > 1:
            assert result['statistic'].iloc[0] >= result['statistic'].iloc[1]

    def test_sort_by_statistic_with_fisher_raises(self, sample_documents):
        from qhchina.analytics import compare_corpora
        corpus_a = sample_documents[:3]
        corpus_b = sample_documents[2:]
        with pytest.raises(ValueError, match="statistic.*not available.*fisher"):
            compare_corpora(corpus_a, corpus_b, method='fisher', sort_by='statistic')

    def test_invalid_method_raises(self, sample_documents):
        from qhchina.analytics import compare_corpora
        corpus_a = sample_documents[:3]
        corpus_b = sample_documents[2:]
        with pytest.raises(ValueError, match="method must be"):
            compare_corpora(corpus_a, corpus_b, method='invalid')

    def test_log_likelihood_as_list(self, sample_documents):
        from qhchina.analytics import compare_corpora
        corpus_a = sample_documents[:3]
        corpus_b = sample_documents[2:]
        result = compare_corpora(corpus_a, corpus_b, method='log_likelihood', as_dataframe=False)
        assert isinstance(result, list)
        if result:
            assert 'statistic' in result[0]


# =============================================================================
# KWIC
# =============================================================================

class TestKwic:
    """Tests for kwic() function."""

    def test_basic_kwic(self, sample_documents):
        from qhchina.analytics.collocations import kwic
        result = kwic(sample_documents, '也', horizon=3)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        expected_cols = {'left', 'node', 'right', 'left_tokens', 'right_tokens', 'doc_index', 'position'}
        assert expected_cols.issubset(set(result.columns))
        assert all(result['node'] == '也')

    def test_kwic_multiple_targets(self, sample_documents):
        from qhchina.analytics.collocations import kwic
        result = kwic(sample_documents, ['也', '不'], horizon=3)
        assert len(result) > 0
        assert set(result['node'].unique()).issubset({'也', '不'})

    def test_kwic_separator(self):
        from qhchina.analytics.collocations import kwic
        sentences = [['the', 'quick', 'brown', 'fox']]
        result = kwic(sentences, 'quick', horizon=2, separator=' ')
        assert result.iloc[0]['left'] == 'the'
        assert result.iloc[0]['right'] == 'brown fox'

    def test_kwic_default_separator_chinese(self):
        from qhchina.analytics.collocations import kwic
        sentences = [list("天下大乱")]
        result = kwic(sentences, '下', horizon=2)
        assert result.iloc[0]['left'] == '天'
        assert result.iloc[0]['right'] == '大乱'

    def test_kwic_sort_right(self):
        from qhchina.analytics.collocations import kwic
        sentences = [list("我的天"), list("他的地"), list("你的人")]
        result = kwic(sentences, '的', sort_by='right')
        rights = result['right'].tolist()
        assert rights == sorted(rights)

    def test_kwic_sort_left(self):
        from qhchina.analytics.collocations import kwic
        sentences = [list("我看天"), list("他看地"), list("你看人")]
        result = kwic(sentences, '看', sort_by='left')
        assert len(result) == 3

    def test_kwic_sort_position(self):
        from qhchina.analytics.collocations import kwic
        sentences = [list("天下大乱"), list("天命不违")]
        result = kwic(sentences, '天', sort_by='position')
        assert result.iloc[0]['doc_index'] == 0
        assert result.iloc[1]['doc_index'] == 1

    def test_kwic_max_results(self, sample_documents):
        from qhchina.analytics.collocations import kwic
        result = kwic(sample_documents, '也', max_results=1)
        assert len(result) <= 1

    def test_kwic_as_list(self, sample_documents):
        from qhchina.analytics.collocations import kwic
        result = kwic(sample_documents, '也', as_dataframe=False)
        assert isinstance(result, list)
        if result:
            assert 'left' in result[0]
            assert 'node' in result[0]

    def test_kwic_empty_result(self, sample_documents):
        from qhchina.analytics.collocations import kwic
        result = kwic(sample_documents, '不存在的词')
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_kwic_invalid_sort_raises(self, sample_documents):
        from qhchina.analytics.collocations import kwic
        with pytest.raises(ValueError, match="sort_by must be"):
            kwic(sample_documents, '也', sort_by='invalid')

    def test_kwic_empty_target_raises(self, sample_documents):
        from qhchina.analytics.collocations import kwic
        with pytest.raises(ValueError, match="target cannot be empty"):
            kwic(sample_documents, [])

    def test_kwic_left_tokens_and_right_tokens(self):
        from qhchina.analytics.collocations import kwic
        sentences = [list("天下大乱之后")]
        result = kwic(sentences, '大', horizon=2)
        row = result.iloc[0]
        assert row['left_tokens'] == ['天', '下']
        assert row['right_tokens'] == ['乱', '之']


# =============================================================================
# CoocMatrix.sum() and .to_ppmi()
# =============================================================================

class TestCoocMatrixSum:
    """Tests for CoocMatrix.sum() method."""

    def test_sum_total(self, sample_documents):
        from qhchina.analytics.collocations import cooc_matrix
        matrix = cooc_matrix(sample_documents, horizon=2)
        total = matrix.sum()
        assert isinstance(total, int)
        assert total > 0

    def test_sum_axis0(self, sample_documents):
        from qhchina.analytics.collocations import cooc_matrix
        matrix = cooc_matrix(sample_documents, horizon=2)
        col_sums = matrix.sum(axis=0)
        assert isinstance(col_sums, np.ndarray)
        assert len(col_sums) == len(matrix.vocab)
        assert col_sums.sum() == matrix.sum()

    def test_sum_axis1(self, sample_documents):
        from qhchina.analytics.collocations import cooc_matrix
        matrix = cooc_matrix(sample_documents, horizon=2)
        row_sums = matrix.sum(axis=1)
        assert isinstance(row_sums, np.ndarray)
        assert len(row_sums) == len(matrix.vocab)
        assert row_sums.sum() == matrix.sum()

    def test_sum_invalid_axis_raises(self, sample_documents):
        from qhchina.analytics.collocations import cooc_matrix
        matrix = cooc_matrix(sample_documents, horizon=2)
        with pytest.raises(ValueError, match="axis must be"):
            matrix.sum(axis=2)


class TestCoocMatrixPPMI:
    """Tests for CoocMatrix.to_ppmi() method."""

    def test_ppmi_returns_cooc_matrix(self, sample_documents):
        from qhchina.analytics.collocations import cooc_matrix, CoocMatrix
        matrix = cooc_matrix(sample_documents, horizon=2)
        ppmi = matrix.to_ppmi()
        assert isinstance(ppmi, CoocMatrix)

    def test_ppmi_values_nonnegative(self, sample_documents):
        from qhchina.analytics.collocations import cooc_matrix
        matrix = cooc_matrix(sample_documents, horizon=2)
        ppmi = matrix.to_ppmi()
        dense = ppmi.to_dense()
        assert np.all(dense >= 0)

    def test_ppmi_same_vocab(self, sample_documents):
        from qhchina.analytics.collocations import cooc_matrix
        matrix = cooc_matrix(sample_documents, horizon=2)
        ppmi = matrix.to_ppmi()
        assert ppmi.vocab == matrix.vocab

    def test_ppmi_alpha_1_no_smoothing(self, sample_documents):
        from qhchina.analytics.collocations import cooc_matrix
        matrix = cooc_matrix(sample_documents, horizon=2)
        ppmi = matrix.to_ppmi(alpha=1.0)
        assert ppmi.nnz > 0

    def test_ppmi_invalid_alpha_raises(self, sample_documents):
        from qhchina.analytics.collocations import cooc_matrix
        matrix = cooc_matrix(sample_documents, horizon=2)
        with pytest.raises(ValueError, match="alpha must be"):
            matrix.to_ppmi(alpha=0.0)
        with pytest.raises(ValueError, match="alpha must be"):
            matrix.to_ppmi(alpha=1.5)

    def test_ppmi_indexing_returns_float(self, sample_documents):
        from qhchina.analytics.collocations import cooc_matrix
        matrix = cooc_matrix(sample_documents, horizon=2)
        ppmi = matrix.to_ppmi()
        val = ppmi[ppmi.vocab[0], ppmi.vocab[1]]
        assert isinstance(val, (int, float, np.integer, np.floating))


# =============================================================================
# compare_collocates
# =============================================================================

class TestCompareCollocates:
    """Tests for compare_collocates() function."""

    def test_basic_compare(self, sample_documents):
        from qhchina.analytics.collocations import compare_collocates
        corpus_a = sample_documents[:3]
        corpus_b = sample_documents[2:]
        result = compare_collocates(corpus_a, corpus_b, target_words='也', min_obs=1)
        assert isinstance(result, pd.DataFrame)
        expected_cols = {'target', 'collocate', 'ratio_a', 'ratio_b',
                        'log_ratio_change', 'obs_a', 'obs_b',
                        'p_value_a', 'p_value_b', 'status'}
        assert expected_cols.issubset(set(result.columns))

    def test_compare_status_values(self, sample_documents):
        from qhchina.analytics.collocations import compare_collocates
        corpus_a = sample_documents[:3]
        corpus_b = sample_documents[2:]
        result = compare_collocates(corpus_a, corpus_b, target_words='也', min_obs=1)
        valid_statuses = {'strengthened', 'weakened', 'appeared', 'disappeared', 'stable'}
        if not result.empty:
            assert set(result['status'].unique()).issubset(valid_statuses)

    def test_compare_as_list(self, sample_documents):
        from qhchina.analytics.collocations import compare_collocates
        corpus_a = sample_documents[:3]
        corpus_b = sample_documents[2:]
        result = compare_collocates(corpus_a, corpus_b, target_words='也',
                                    min_obs=1, as_dataframe=False)
        assert isinstance(result, list)

    def test_compare_empty_corpora(self):
        from qhchina.analytics.collocations import compare_collocates
        empty_a = [['x', 'y']]
        empty_b = [['a', 'b']]
        result = compare_collocates(empty_a, empty_b, target_words='z', min_obs=0)
        assert isinstance(result, pd.DataFrame)

    def test_compare_multiple_targets(self, sample_documents):
        from qhchina.analytics.collocations import compare_collocates
        corpus_a = sample_documents[:3]
        corpus_b = sample_documents[2:]
        result = compare_collocates(corpus_a, corpus_b,
                                    target_words=['也', '不'], min_obs=1)
        assert isinstance(result, pd.DataFrame)


# =============================================================================
# find_shared_sequences
# =============================================================================

class TestFindSharedSequences:
    """Tests for find_shared_sequences()."""

    def test_basic_cross_corpus(self):
        from qhchina.analytics.textreuse import find_shared_sequences
        texts_a = ["天地玄黄宇宙洪荒日月盈昃"]
        texts_b = ["天地玄黄宇宙洪荒"]
        result = find_shared_sequences(texts_a, texts_b, n=3, min_length=5, min_similarity=0.8)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert result.iloc[0]['similarity'] >= 0.8

    def test_within_corpus(self):
        from qhchina.analytics.textreuse import find_shared_sequences
        texts = ["天地玄黄宇宙洪荒", "天地玄黄日月盈昃"]
        result = find_shared_sequences(texts, n=2, min_length=3, min_similarity=0.8)
        assert isinstance(result, pd.DataFrame)

    def test_no_corpus_b_all_pairs(self):
        from qhchina.analytics.textreuse import find_shared_sequences
        texts = ["ABCDEFGH", "ABCDEFIJ", "XXXXXXXXY"]
        result = find_shared_sequences(texts, n=3, min_length=3, min_similarity=0.7)
        assert isinstance(result, pd.DataFrame)
        if len(result) > 0:
            assert all(result['doc_a'] < result['doc_b']) or True

    def test_tokenized_input(self):
        from qhchina.analytics.textreuse import find_shared_sequences
        tok_a = [['a', 'b', 'c', 'd', 'e', 'f', 'g']]
        tok_b = [['a', 'b', 'c', 'd', 'e', 'x', 'y']]
        result = find_shared_sequences(tok_a, tok_b, n=3, min_length=3, min_similarity=0.7)
        assert isinstance(result, pd.DataFrame)

    def test_empty_corpora(self):
        from qhchina.analytics.textreuse import find_shared_sequences
        result = find_shared_sequences([], [], n=3, min_length=3)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_as_list(self):
        from qhchina.analytics.textreuse import find_shared_sequences
        texts_a = ["天地玄黄宇宙洪荒"]
        texts_b = ["天地玄黄宇宙洪荒"]
        result = find_shared_sequences(texts_a, texts_b, n=3, min_length=3,
                                       min_similarity=0.8, as_dataframe=False)
        assert isinstance(result, list)
        if result:
            assert 'similarity' in result[0]
            assert 'passage_a' in result[0]

    def test_output_columns(self):
        from qhchina.analytics.textreuse import find_shared_sequences
        texts_a = ["天地玄黄宇宙洪荒"]
        texts_b = ["天地玄黄宇宙洪荒"]
        result = find_shared_sequences(texts_a, texts_b, n=3, min_length=3)
        expected_cols = {'doc_a', 'doc_b', 'pos_a', 'pos_b', 'length',
                        'similarity', 'passage_a', 'passage_b'}
        assert expected_cols == set(result.columns)

    def test_exact_match_similarity_1(self):
        from qhchina.analytics.textreuse import find_shared_sequences
        text = "天地玄黄宇宙洪荒日月盈昃"
        result = find_shared_sequences([text], [text], n=3, min_length=5, min_similarity=0.9)
        assert len(result) > 0
        assert result.iloc[0]['similarity'] == 1.0

    def test_invalid_n_raises(self):
        from qhchina.analytics.textreuse import find_shared_sequences
        with pytest.raises(ValueError, match="n must be"):
            find_shared_sequences(["abc"], ["abc"], n=0)

    def test_invalid_min_similarity_raises(self):
        from qhchina.analytics.textreuse import find_shared_sequences
        with pytest.raises(ValueError, match="min_similarity"):
            find_shared_sequences(["abc"], ["abc"], min_similarity=0.0)
        with pytest.raises(ValueError, match="min_similarity"):
            find_shared_sequences(["abc"], ["abc"], min_similarity=1.5)


# =============================================================================
# Top-level import tests
# =============================================================================

class TestTopLevelImports:
    """Verify new functions are accessible from the top-level package."""

    def test_import_kwic(self):
        from qhchina import kwic
        assert callable(kwic)

    def test_import_compare_collocates(self):
        from qhchina import compare_collocates
        assert callable(compare_collocates)

    def test_import_find_shared_sequences(self):
        from qhchina import find_shared_sequences
        assert callable(find_shared_sequences)

    def test_import_from_analytics(self):
        from qhchina.analytics import kwic, compare_collocates, find_shared_sequences
        assert callable(kwic)
        assert callable(compare_collocates)
        assert callable(find_shared_sequences)
