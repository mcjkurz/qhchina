"""
Tests for qhchina.analytics.textreuse module.
"""
import pytest
import pandas as pd


class TestFindSharedSequences:
    """Tests for find_shared_sequences()."""

    def test_basic_match(self):
        from qhchina.analytics.textreuse import find_shared_sequences
        docs = [list("天地玄黄宇宙洪荒日月盈昃"), list("天地玄黄宇宙洪荒")]
        result = find_shared_sequences(docs, n=3, min_length=5, min_similarity=0.8)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert result.iloc[0]['similarity'] >= 0.8

    def test_multiple_documents(self):
        from qhchina.analytics.textreuse import find_shared_sequences
        docs = [list("天地玄黄宇宙洪荒"), list("天地玄黄日月盈昃")]
        result = find_shared_sequences(docs, n=2, min_length=3, min_similarity=0.8)
        assert isinstance(result, pd.DataFrame)

    def test_all_pairs(self):
        from qhchina.analytics.textreuse import find_shared_sequences
        docs = [list("ABCDEFGH"), list("ABCDEFIJ"), list("XXXXXXXXY")]
        result = find_shared_sequences(docs, n=3, min_length=3, min_similarity=0.7)
        assert isinstance(result, pd.DataFrame)
        if len(result) > 0:
            assert all(result['doc_a'] <= result['doc_b'])

    def test_tokenized_input(self):
        from qhchina.analytics.textreuse import find_shared_sequences
        docs = [['a', 'b', 'c', 'd', 'e', 'f', 'g'],
                ['a', 'b', 'c', 'd', 'e', 'x', 'y']]
        result = find_shared_sequences(docs, n=3, min_length=3, min_similarity=0.7)
        assert isinstance(result, pd.DataFrame)

    def test_empty_documents(self):
        from qhchina.analytics.textreuse import find_shared_sequences
        result = find_shared_sequences([], n=3, min_length=3)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_as_list(self):
        from qhchina.analytics.textreuse import find_shared_sequences
        docs = [list("天地玄黄宇宙洪荒"), list("天地玄黄宇宙洪荒")]
        result = find_shared_sequences(docs, n=3, min_length=3,
                                       min_similarity=0.8, as_dataframe=False)
        assert isinstance(result, list)
        if result:
            assert 'similarity' in result[0]
            assert 'passage_a' in result[0]

    def test_output_columns(self):
        from qhchina.analytics.textreuse import find_shared_sequences
        docs = [list("天地玄黄宇宙洪荒"), list("天地玄黄宇宙洪荒")]
        result = find_shared_sequences(docs, n=3, min_length=3)
        expected_cols = {'doc_a', 'doc_b', 'pos_a', 'pos_b', 'length',
                        'similarity', 'passage_a', 'passage_b'}
        assert expected_cols == set(result.columns)

    def test_exact_match_similarity_1(self):
        from qhchina.analytics.textreuse import find_shared_sequences
        doc = list("天地玄黄宇宙洪荒日月盈昃")
        result = find_shared_sequences([doc, doc], n=3, min_length=5, min_similarity=0.9)
        assert len(result) > 0
        assert result.iloc[0]['similarity'] == 1.0

    def test_within_documents_false_skips_self(self):
        from qhchina.analytics.textreuse import find_shared_sequences
        doc = list("ABCDEFABCDEF")
        result = find_shared_sequences([doc], n=3, min_length=3,
                                       min_similarity=0.8, within_documents=False)
        assert len(result) == 0

    def test_within_documents_true_finds_self_reuse(self):
        from qhchina.analytics.textreuse import find_shared_sequences
        doc = list("ABCDEFGHIJ") + list("ABCDEFGHIJ")
        result = find_shared_sequences([doc], n=3, min_length=5,
                                       min_similarity=0.8, within_documents=True)
        assert len(result) > 0
        assert result.iloc[0]['doc_a'] == 0
        assert result.iloc[0]['doc_b'] == 0

    def test_invalid_n_raises(self):
        from qhchina.analytics.textreuse import find_shared_sequences
        with pytest.raises(ValueError, match="n must be"):
            find_shared_sequences([list("abc")], n=0)

    def test_invalid_min_similarity_raises(self):
        from qhchina.analytics.textreuse import find_shared_sequences
        with pytest.raises(ValueError, match="min_similarity"):
            find_shared_sequences([list("abc")], min_similarity=0.0)
        with pytest.raises(ValueError, match="min_similarity"):
            find_shared_sequences([list("abc")], min_similarity=1.5)

    def test_rejects_raw_string(self):
        from qhchina.analytics.textreuse import find_shared_sequences
        with pytest.raises(TypeError, match="documents must be"):
            find_shared_sequences("天地玄黄")

    def test_rejects_list_of_strings(self):
        from qhchina.analytics.textreuse import find_shared_sequences
        with pytest.raises(TypeError, match="Each document must be"):
            find_shared_sequences(["天地玄黄", "宇宙洪荒"])
