"""Tests for the Corpus class."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from qhchina import Corpus
from qhchina.corpus import Document


class TestDocument:
    """Tests for the Document dataclass."""
    
    def test_document_creation(self):
        """Test basic document creation."""
        doc = Document(tokens=['word1', 'word2'], metadata={'author': 'Test'}, doc_id='doc_0')
        assert doc.tokens == ['word1', 'word2']
        assert doc.metadata == {'author': 'Test'}
        assert doc.doc_id == 'doc_0'
    
    def test_document_len(self):
        """Test document length."""
        doc = Document(tokens=['a', 'b', 'c'])
        assert len(doc) == 3
    
    def test_document_iter(self):
        """Test document iteration."""
        doc = Document(tokens=['a', 'b', 'c'])
        assert list(doc) == ['a', 'b', 'c']
    
    def test_document_empty_metadata(self):
        """Test document with no metadata."""
        doc = Document(tokens=['word'])
        assert doc.metadata == {}
        assert doc.doc_id == ''
    
    def test_document_getitem_int(self):
        """Test token access by integer index."""
        doc = Document(tokens=['a', 'b', 'c'], metadata={'author': 'Test'})
        assert doc[0] == 'a'
        assert doc[1] == 'b'
        assert doc[-1] == 'c'
    
    def test_document_getitem_str(self):
        """Test metadata access by string key."""
        doc = Document(tokens=['a', 'b'], metadata={'author': 'Test', 'year': 1920})
        assert doc["author"] == 'Test'
        assert doc["year"] == 1920
    
    def test_document_contains(self):
        """Test metadata key containment check."""
        doc = Document(tokens=['a'], metadata={'author': 'Test'})
        assert "author" in doc
        assert "year" not in doc
    
    def test_document_get_with_default(self):
        """Test metadata get with default value."""
        doc = Document(tokens=['a'], metadata={'author': 'Test'})
        assert doc.get("author") == 'Test'
        assert doc.get("year") is None
        assert doc.get("year", 1900) == 1900


class TestCorpusCreation:
    """Tests for Corpus creation and initialization."""
    
    def test_empty_corpus(self):
        """Test creating an empty corpus."""
        corpus = Corpus()
        assert len(corpus) == 0
        assert corpus.token_count == 0
        assert not corpus
    
    def test_corpus_from_token_lists(self):
        """Test creating corpus from list of token lists."""
        docs = [['word1', 'word2'], ['word3']]
        corpus = Corpus(docs)
        assert len(corpus) == 2
        assert corpus.token_count == 3
    
    def test_corpus_from_documents(self):
        """Test creating corpus from Document objects."""
        docs = [
            Document(tokens=['a', 'b'], metadata={'x': 1}),
            Document(tokens=['c'], metadata={'x': 2}),
        ]
        corpus = Corpus(docs)
        assert len(corpus) == 2
        # Metadata should be preserved
        assert corpus[0].metadata == {'x': 1}
    
    def test_corpus_from_corpus(self):
        """Test creating corpus from another corpus."""
        original = Corpus([['a', 'b'], ['c']])
        original.add(['d'], author='test')
        
        copy = Corpus(original)
        assert len(copy) == 3
        # Should be independent (new doc_ids)
        assert 'doc_0' in copy
    
    def test_corpus_with_default_metadata(self):
        """Test corpus with default metadata."""
        corpus = Corpus(metadata={'source': 'test_source'})
        corpus.add(['word1'])
        corpus.add(['word2'], author='Author1')
        
        assert corpus[0].metadata['source'] == 'test_source'
        assert corpus[1].metadata['source'] == 'test_source'
        assert corpus[1].metadata['author'] == 'Author1'


class TestCorpusIteration:
    """Tests for Corpus iteration."""
    
    def test_iter_yields_token_lists(self):
        """Test that iteration yields token lists directly."""
        corpus = Corpus([['a', 'b'], ['c', 'd', 'e']])
        result = list(corpus)
        assert result == [['a', 'b'], ['c', 'd', 'e']]
    
    def test_iter_yields_references(self):
        """Test that iteration yields references, not copies."""
        tokens = ['a', 'b']
        corpus = Corpus([tokens])
        for doc_tokens in corpus:
            assert doc_tokens is tokens
    
    def test_corpus_bool(self):
        """Test corpus truthiness."""
        empty = Corpus()
        assert not empty
        
        non_empty = Corpus([['a']])
        assert non_empty


class TestCorpusDocumentManagement:
    """Tests for adding, getting, and removing documents."""
    
    def test_add_returns_doc_id(self):
        """Test that add returns document ID."""
        corpus = Corpus()
        doc_id = corpus.add(['word1', 'word2'])
        assert doc_id == 'doc_0'
        
        doc_id2 = corpus.add(['word3'])
        assert doc_id2 == 'doc_1'
    
    def test_add_with_custom_doc_id(self):
        """Test adding with custom doc_id."""
        corpus = Corpus()
        doc_id = corpus.add(['word'], doc_id='my_custom_id')
        assert doc_id == 'my_custom_id'
        assert 'my_custom_id' in corpus
    
    def test_add_duplicate_doc_id_raises(self):
        """Test that duplicate doc_id raises error."""
        corpus = Corpus()
        corpus.add(['word1'], doc_id='dup')
        with pytest.raises(ValueError, match="already exists"):
            corpus.add(['word2'], doc_id='dup')
    
    def test_add_with_metadata(self):
        """Test adding with metadata."""
        corpus = Corpus()
        corpus.add(['word'], author='Author1', year=2020)
        
        doc = corpus[0]
        assert doc.metadata['author'] == 'Author1'
        assert doc.metadata['year'] == 2020
    
    def test_add_invalid_tokens_raises(self):
        """Test that non-list tokens raise error."""
        corpus = Corpus()
        with pytest.raises(TypeError, match="must be a list"):
            corpus.add("not a list")
    
    def test_add_many(self):
        """Test adding multiple documents."""
        corpus = Corpus()
        doc_ids = corpus.add_many([['a'], ['b'], ['c']])
        assert len(doc_ids) == 3
        assert len(corpus) == 3
    
    def test_add_many_with_shared_metadata(self):
        """Test add_many with shared metadata."""
        corpus = Corpus()
        corpus.add_many([['a'], ['b']], source='test')
        
        assert corpus[0].metadata['source'] == 'test'
        assert corpus[1].metadata['source'] == 'test'
    
    def test_add_many_with_per_doc_metadata(self):
        """Test add_many with per-document metadata."""
        corpus = Corpus()
        metadata_list = [{'author': 'A'}, {'author': 'B'}]
        corpus.add_many([['a'], ['b']], metadata_list=metadata_list)
        
        assert corpus[0].metadata['author'] == 'A'
        assert corpus[1].metadata['author'] == 'B'
    
    def test_add_many_metadata_length_mismatch(self):
        """Test that mismatched metadata_list raises error."""
        corpus = Corpus()
        with pytest.raises(ValueError, match="must match"):
            corpus.add_many([['a'], ['b']], metadata_list=[{'x': 1}])
    
    def test_get_by_doc_id(self):
        """Test getting document by ID."""
        corpus = Corpus()
        corpus.add(['word'], doc_id='my_id')
        
        doc = corpus.get('my_id')
        assert doc.tokens == ['word']
    
    def test_get_nonexistent_raises(self):
        """Test that getting nonexistent doc raises error."""
        corpus = Corpus()
        with pytest.raises(KeyError, match="not found"):
            corpus.get('nonexistent')
    
    def test_getitem_by_index(self):
        """Test getting document by index."""
        corpus = Corpus([['a'], ['b'], ['c']])
        assert corpus[1].tokens == ['b']
    
    def test_getitem_by_doc_id(self):
        """Test getting document by doc_id string."""
        corpus = Corpus()
        corpus.add(['word'], doc_id='my_id')
        assert corpus['my_id'].tokens == ['word']
    
    def test_getitem_slice(self):
        """Test slicing corpus."""
        corpus = Corpus([['a'], ['b'], ['c'], ['d']])
        sliced = corpus[1:3]
        assert isinstance(sliced, Corpus)
        assert len(sliced) == 2
        assert list(sliced) == [['b'], ['c']]
    
    def test_contains(self):
        """Test 'in' operator."""
        corpus = Corpus()
        corpus.add(['word'], doc_id='exists')
        
        assert 'exists' in corpus
        assert 'not_exists' not in corpus
    
    def test_remove(self):
        """Test removing document."""
        corpus = Corpus()
        corpus.add(['a'], doc_id='to_remove')
        corpus.add(['b'], doc_id='to_keep')
        
        removed = corpus.remove('to_remove')
        assert removed.tokens == ['a']
        assert len(corpus) == 1
        assert 'to_remove' not in corpus
        assert 'to_keep' in corpus
    
    def test_remove_nonexistent_raises(self):
        """Test removing nonexistent doc raises error."""
        corpus = Corpus()
        with pytest.raises(KeyError, match="not found"):
            corpus.remove('nonexistent')


class TestCorpusFilter:
    """Tests for corpus filtering."""
    
    def test_filter_by_metadata(self):
        """Test filtering by metadata value."""
        corpus = Corpus()
        corpus.add(['a'], author='A')
        corpus.add(['b'], author='B')
        corpus.add(['c'], author='A')
        
        filtered = corpus.filter(author='A')
        assert len(filtered) == 2
        assert list(filtered) == [['a'], ['c']]
    
    def test_filter_by_predicate(self):
        """Test filtering by predicate function."""
        corpus = Corpus([['a'], ['b', 'c', 'd'], ['e', 'f']])
        
        long_docs = corpus.filter(lambda d: len(d.tokens) > 1)
        assert len(long_docs) == 2
    
    def test_filter_combined(self):
        """Test filtering with both predicate and metadata."""
        corpus = Corpus()
        corpus.add(['a', 'b'], author='A')
        corpus.add(['c'], author='A')
        corpus.add(['d', 'e', 'f'], author='B')
        
        result = corpus.filter(
            lambda d: len(d.tokens) > 1,
            author='A'
        )
        assert len(result) == 1
        assert result[0].tokens == ['a', 'b']
    
    def test_filter_returns_view(self):
        """Test that filter returns view with shared references."""
        corpus = Corpus()
        tokens = ['a', 'b']
        corpus.add(tokens, author='A')
        
        filtered = corpus.filter(author='A')
        # Should share the same token list
        assert filtered[0].tokens is tokens
    
    def test_filter_no_matches(self):
        """Test filter with no matches."""
        corpus = Corpus([['a']])
        corpus[0].metadata['author'] = 'A'
        
        filtered = corpus.filter(author='B')
        assert len(filtered) == 0


class TestCorpusGroupby:
    """Tests for corpus groupby."""
    
    def test_groupby_basic(self):
        """Test basic groupby."""
        corpus = Corpus()
        corpus.add(['a'], author='Author1')
        corpus.add(['b'], author='Author1')
        corpus.add(['c'], author='Author2')
        
        grouped = corpus.groupby('author')
        
        assert set(grouped.keys()) == {'Author1', 'Author2'}
        assert grouped['Author1'] == [['a'], ['b']]
        assert grouped['Author2'] == [['c']]
    
    def test_groupby_returns_references(self):
        """Test that groupby returns token list references."""
        corpus = Corpus()
        tokens = ['a', 'b']
        corpus.add(tokens, author='A')
        
        grouped = corpus.groupby('author')
        assert grouped['A'][0] is tokens
    
    def test_groupby_skips_missing_key(self):
        """Test that groupby skips documents without the key."""
        corpus = Corpus()
        corpus.add(['a'], author='A')
        corpus.add(['b'])  # No author
        corpus.add(['c'], author='A')
        
        grouped = corpus.groupby('author')
        assert grouped == {'A': [['a'], ['c']]}
    
    def test_groupby_requires_string_key(self):
        """Test that groupby requires string key."""
        corpus = Corpus()
        with pytest.raises(TypeError, match="must be a string"):
            corpus.groupby(123)
    
    def test_groupby_empty_corpus(self):
        """Test groupby on empty corpus."""
        corpus = Corpus()
        grouped = corpus.groupby('author')
        assert grouped == {}


class TestCorpusSplit:
    """Tests for corpus splitting."""
    
    def test_split_basic(self):
        """Test basic train/test split."""
        corpus = Corpus([['a'], ['b'], ['c'], ['d'], ['e']])
        train, test = corpus.split(0.6, seed=42)
        
        assert len(train) == 3
        assert len(test) == 2
        assert len(train) + len(test) == len(corpus)
    
    def test_split_reproducible(self):
        """Test that split is reproducible with seed."""
        corpus = Corpus([['a'], ['b'], ['c'], ['d'], ['e']])
        
        train1, test1 = corpus.split(0.6, seed=42)
        train2, test2 = corpus.split(0.6, seed=42)
        
        assert list(train1) == list(train2)
        assert list(test1) == list(test2)
    
    def test_split_stratified(self):
        """Test stratified split."""
        corpus = Corpus()
        # 4 docs from author A, 4 from author B
        for _ in range(4):
            corpus.add(['a'], author='A')
        for _ in range(4):
            corpus.add(['b'], author='B')
        
        train, test = corpus.split(0.5, stratify_by='author', seed=42)
        
        # Each group should be split roughly 50/50
        train_a = sum(1 for d in train._documents if d.metadata.get('author') == 'A')
        train_b = sum(1 for d in train._documents if d.metadata.get('author') == 'B')
        
        assert train_a == 2
        assert train_b == 2
    
    def test_split_invalid_ratio(self):
        """Test that invalid ratio raises error."""
        corpus = Corpus([['a']])
        
        with pytest.raises(ValueError, match="between 0 and 1"):
            corpus.split(0.0)
        with pytest.raises(ValueError, match="between 0 and 1"):
            corpus.split(1.0)
        with pytest.raises(ValueError, match="between 0 and 1"):
            corpus.split(1.5)


class TestCorpusStatistics:
    """Tests for corpus statistics."""
    
    def test_token_count(self):
        """Test token count."""
        corpus = Corpus([['a', 'b'], ['c', 'd', 'e']])
        assert corpus.token_count == 5
    
    def test_vocab(self):
        """Test vocabulary."""
        corpus = Corpus([['a', 'b', 'a'], ['b', 'c']])
        vocab = corpus.vocab
        
        assert vocab['a'] == 2
        assert vocab['b'] == 2
        assert vocab['c'] == 1
    
    def test_vocab_size(self):
        """Test vocabulary size."""
        corpus = Corpus([['a', 'b', 'a'], ['b', 'c']])
        assert corpus.vocab_size == 3
    
    def test_metadata_keys(self):
        """Test metadata_keys property."""
        corpus = Corpus()
        corpus.add(['a'], author='A', year=2020)
        corpus.add(['b'], author='B', source='test')
        
        keys = corpus.metadata_keys
        assert keys == {'author', 'year', 'source'}
    
    def test_metadata_values(self):
        """Test metadata_values method."""
        corpus = Corpus()
        corpus.add(['a'], author='A')
        corpus.add(['b'], author='B')
        corpus.add(['c'], author='A')
        corpus.add(['d'])  # No author
        
        values = corpus.metadata_values('author')
        assert values == {'A', 'B'}
    
    def test_describe(self):
        """Test describe method."""
        corpus = Corpus([['a', 'b'], ['c']])
        desc = corpus.describe()
        
        assert desc['documents'] == 2
        assert desc['tokens'] == 3
        assert desc['vocab_size'] == 3
        assert desc['avg_doc_length'] == 1.5
    
    def test_statistics_cached(self):
        """Test that statistics are cached."""
        corpus = Corpus([['a', 'b'], ['c']])
        
        # Access twice
        _ = corpus.token_count
        _ = corpus.token_count
        
        # Cache should exist
        assert corpus._token_count_cache is not None
    
    def test_cache_invalidated_on_add(self):
        """Test that cache is invalidated when adding."""
        corpus = Corpus([['a']])
        _ = corpus.token_count
        
        corpus.add(['b', 'c'])
        assert corpus._token_count_cache is None
        assert corpus.token_count == 3


class TestCorpusSerialization:
    """Tests for corpus save/load."""
    
    def test_save_load_json_roundtrip(self):
        """Test save and load roundtrip with JSON format."""
        corpus = Corpus()
        corpus.add(['a', 'b'], author='A', year=2020)
        corpus.add(['c'], author='B')
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        
        try:
            corpus.save(path)
            loaded = Corpus.load(path)
            
            assert len(loaded) == 2
            assert list(loaded) == [['a', 'b'], ['c']]
            assert loaded[0].metadata['author'] == 'A'
            assert loaded[0].metadata['year'] == 2020
        finally:
            Path(path).unlink()
    
    def test_save_load_pickle_roundtrip(self):
        """Test save and load roundtrip with pickle format."""
        corpus = Corpus()
        corpus.add(['a', 'b'], author='A', year=2020)
        corpus.add(['c'], author='B')
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name
        
        try:
            corpus.save(path)
            loaded = Corpus.load(path)
            
            assert len(loaded) == 2
            assert list(loaded) == [['a', 'b'], ['c']]
            assert loaded[0].metadata['author'] == 'A'
            assert loaded[0].metadata['year'] == 2020
        finally:
            Path(path).unlink()
    
    def test_save_load_explicit_format(self):
        """Test save and load with explicit format parameter."""
        corpus = Corpus()
        corpus.add(['a', 'b'], author='A')
        
        with tempfile.NamedTemporaryFile(suffix='.data', delete=False) as f:
            path = f.name
        
        try:
            # Save with explicit format (extension doesn't match)
            corpus.save(path, format='pickle')
            loaded = Corpus.load(path, format='pickle')
            
            assert len(loaded) == 1
            assert list(loaded) == [['a', 'b']]
        finally:
            Path(path).unlink()
    
    def test_save_unknown_extension_raises(self):
        """Test that unknown extension without format raises error."""
        corpus = Corpus([['a']])
        
        with tempfile.NamedTemporaryFile(suffix='.unknown', delete=False) as f:
            path = f.name
        
        try:
            with pytest.raises(ValueError, match="Cannot infer format"):
                corpus.save(path)
        finally:
            Path(path).unlink(missing_ok=True)
    
    def test_load_unknown_extension_raises(self):
        """Test that loading unknown extension without format raises error."""
        with tempfile.NamedTemporaryFile(suffix='.unknown', delete=False) as f:
            path = f.name
        
        try:
            with pytest.raises(ValueError, match="Cannot infer format"):
                Corpus.load(path)
        finally:
            Path(path).unlink(missing_ok=True)
    
    def test_save_invalid_format_raises(self):
        """Test that invalid format raises error."""
        corpus = Corpus([['a']])
        
        with pytest.raises(ValueError, match="must be 'json' or 'pickle'"):
            corpus.save('test.json', format='xml')
    
    def test_save_load_preserves_doc_ids(self):
        """Test that save/load preserves doc_ids."""
        corpus = Corpus()
        corpus.add(['a'], doc_id='custom_id')
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        
        try:
            corpus.save(path)
            loaded = Corpus.load(path)
            
            assert 'custom_id' in loaded
        finally:
            Path(path).unlink()
    
    def test_pickle_extension_variants(self):
        """Test that both .pkl and .pickle extensions work."""
        corpus = Corpus([['a', 'b']])
        
        # Test .pkl
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path_pkl = f.name
        
        # Test .pickle
        with tempfile.NamedTemporaryFile(suffix='.pickle', delete=False) as f:
            path_pickle = f.name
        
        try:
            corpus.save(path_pkl)
            corpus.save(path_pickle)
            
            loaded_pkl = Corpus.load(path_pkl)
            loaded_pickle = Corpus.load(path_pickle)
            
            assert list(loaded_pkl) == [['a', 'b']]
            assert list(loaded_pickle) == [['a', 'b']]
        finally:
            Path(path_pkl).unlink()
            Path(path_pickle).unlink()
    
    def test_to_dataframe(self):
        """Test conversion to DataFrame."""
        corpus = Corpus()
        corpus.add(['a', 'b'], author='A')
        corpus.add(['c'], author='B')
        
        df = corpus.to_dataframe()
        
        assert len(df) == 2
        assert 'doc_id' in df.columns
        assert 'tokens' in df.columns
        assert 'token_count' in df.columns
        assert 'author' in df.columns
        assert df['token_count'].tolist() == [2, 1]


class TestCorpusIntegration:
    """Integration tests with analytics modules."""
    
    def test_integration_with_lda(self):
        """Test that Corpus works with LDAGibbsSampler."""
        from qhchina.analytics.topicmodels import LDAGibbsSampler
        
        # Create corpus with enough data
        corpus = Corpus()
        for i in range(20):
            corpus.add(['word1', 'word2', 'word3', 'word4'] * 10)
        
        lda = LDAGibbsSampler(n_topics=2, iterations=5, random_state=42)
        lda.fit(corpus)  # Should work directly
        
        assert lda.phi is not None
    
    def test_integration_with_find_collocates(self):
        """Test that Corpus works with find_collocates."""
        from qhchina.analytics.collocations import find_collocates
        
        corpus = Corpus()
        for _ in range(10):
            corpus.add(['the', 'quick', 'brown', 'fox', 'jumps'])
            corpus.add(['the', 'lazy', 'dog', 'sleeps'])
        
        result = find_collocates(corpus, target_words=['the'])
        
        assert len(result) > 0
    
    def test_integration_with_cooc_matrix(self):
        """Test that Corpus works with cooc_matrix."""
        from qhchina.analytics.collocations import cooc_matrix
        
        corpus = Corpus([
            ['a', 'b', 'c'],
            ['b', 'c', 'd'],
            ['a', 'c', 'd'],
        ])
        
        matrix = cooc_matrix(corpus, horizon=2)
        
        assert 'a' in matrix
        assert matrix['a', 'b'] >= 0
    
    def test_integration_groupby_with_stylometry(self):
        """Test that groupby output works with Stylometry format."""
        corpus = Corpus()
        corpus.add(['word1', 'word2'] * 50, author='A')
        corpus.add(['word3', 'word4'] * 50, author='A')
        corpus.add(['word1', 'word3'] * 50, author='B')
        
        grouped = corpus.groupby('author')
        
        # Should be in Stylometry format: dict[str, list[list[str]]]
        assert isinstance(grouped, dict)
        assert all(isinstance(v, list) for v in grouped.values())
        assert all(isinstance(doc, list) for docs in grouped.values() for doc in docs)
        
        # Check structure
        assert 'A' in grouped
        assert 'B' in grouped
        assert len(grouped['A']) == 2
        assert len(grouped['B']) == 1


class TestCorpusRepr:
    """Tests for corpus string representation."""
    
    def test_repr(self):
        """Test repr output."""
        corpus = Corpus([['a', 'b'], ['c']])
        repr_str = repr(corpus)
        
        assert 'Corpus' in repr_str
        assert 'documents=2' in repr_str
        assert 'tokens=3' in repr_str
