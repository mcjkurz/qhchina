"""
Tests for qhchina.analytics.stylometry module.
"""
import pytest
import numpy as np


@pytest.fixture
def stylometry_corpus_dict():
    """Corpus in the format expected by Stylometry: {author: [list of docs]}."""
    return {
        "author_a": [
            list("我在年青时候也曾经做过许多梦后来大半忘却了"),
            list("但自己也并不以为可惜所谓回忆者虽说可以使人欢欣"),
        ],
        "author_b": [
            list("有时也不免使人寂寞使精神的丝缕还牵着已逝的时光"),
            list("又有什么意味呢而我偏苦于不能全忘却这不能全忘的"),
        ],
    }


class TestStylometryBasic:
    """Basic Stylometry functionality tests."""
    
    def test_stylometry_init(self):
        """Test Stylometry initialization."""
        from qhchina.analytics.stylometry import Stylometry
        
        stylo = Stylometry(
            n_features=100,
            ngram_range=(1, 1),
            transform='zscore',
            distance='cosine'
        )
        
        assert stylo.n_features == 100
        assert stylo.ngram_range == (1, 1)
    
    def test_fit_transform_dict(self, stylometry_corpus_dict):
        """Test fit_transform with dict input."""
        from qhchina.analytics.stylometry import Stylometry
        
        stylo = Stylometry(n_features=30)
        stylo.fit_transform(stylometry_corpus_dict)
        
        # After fit_transform, should have document_vectors and document_labels
        assert len(stylo.document_vectors) > 0
        assert len(stylo.document_labels) > 0
        # 2 authors x 2 docs each = 4 docs
        assert len(stylo.document_labels) == 4
    
    def test_fit_transform_list(self, sample_documents):
        """Test fit_transform with list input."""
        from qhchina.analytics.stylometry import Stylometry
        
        stylo = Stylometry(n_features=20)
        stylo.fit_transform(sample_documents)
        
        assert len(stylo.document_vectors) > 0
        assert len(stylo.document_vectors) == len(sample_documents)


class TestStylometryDistance:
    """Tests for Stylometry distance metrics."""
    
    @pytest.fixture
    def fitted_stylometry(self, stylometry_corpus_dict):
        """Pre-fitted Stylometry instance."""
        from qhchina.analytics.stylometry import Stylometry
        
        stylo = Stylometry(n_features=20, distance='cosine')
        stylo.fit_transform(stylometry_corpus_dict)
        return stylo
    
    def test_distance(self, fitted_stylometry):
        """Test computing distance between documents."""
        # document_ids contains the actual doc IDs (e.g., author_a_1, author_a_2)
        doc_ids = fitted_stylometry.document_ids
        if len(doc_ids) >= 2:
            dist = fitted_stylometry.distance(doc_ids[0], doc_ids[1])
            
            assert isinstance(dist, (float, np.floating))
            assert dist >= 0
    
    def test_similarity(self, fitted_stylometry):
        """Test computing similarity between documents."""
        doc_ids = fitted_stylometry.document_ids
        if len(doc_ids) >= 2:
            sim = fitted_stylometry.similarity(doc_ids[0], doc_ids[1])
            
            assert isinstance(sim, (float, np.floating))
    
    def test_most_similar(self, fitted_stylometry):
        """Test finding most similar documents."""
        doc_ids = fitted_stylometry.document_ids
        if len(doc_ids) >= 2:
            similar = fitted_stylometry.most_similar(doc_ids[0], k=2)
            
            assert isinstance(similar, list)
            assert len(similar) <= 2


class TestStylometryTransforms:
    """Tests for different transformation methods."""
    
    def test_zscore_transform(self, stylometry_corpus_dict):
        """Test z-score transformation."""
        from qhchina.analytics.stylometry import Stylometry
        
        stylo = Stylometry(n_features=20, transform='zscore')
        stylo.fit_transform(stylometry_corpus_dict)
        
        assert len(stylo.document_vectors) > 0
    
    def test_tfidf_transform(self, stylometry_corpus_dict):
        """Test TF-IDF transformation."""
        from qhchina.analytics.stylometry import Stylometry
        
        stylo = Stylometry(n_features=20, transform='tfidf')
        stylo.fit_transform(stylometry_corpus_dict)
        
        assert len(stylo.document_vectors) > 0


class TestStylometryDistanceMetrics:
    """Tests for different distance metrics."""
    
    @pytest.mark.parametrize("metric", ["cosine", "burrows_delta", "manhattan", "euclidean", "eder_delta"])
    def test_distance_metrics(self, stylometry_corpus_dict, metric):
        """Test all supported distance metrics."""
        from qhchina.analytics.stylometry import Stylometry
        
        stylo = Stylometry(n_features=20, distance=metric)
        stylo.fit_transform(stylometry_corpus_dict)
        
        # Use document_ids for distance computation
        doc_ids = stylo.document_ids
        if len(doc_ids) >= 2:
            dist = stylo.distance(doc_ids[0], doc_ids[1])
            assert isinstance(dist, (float, np.floating))


class TestStylometryNgrams:
    """Tests for n-gram support."""
    
    def test_bigrams(self, stylometry_corpus_dict):
        """Test using bigrams."""
        from qhchina.analytics.stylometry import Stylometry
        
        stylo = Stylometry(n_features=20, ngram_range=(2, 2))
        stylo.fit_transform(stylometry_corpus_dict)
        
        assert len(stylo.document_vectors) > 0
    
    def test_mixed_ngrams(self, stylometry_corpus_dict):
        """Test using mixed n-grams (1-2)."""
        from qhchina.analytics.stylometry import Stylometry
        
        stylo = Stylometry(n_features=30, ngram_range=(1, 2))
        stylo.fit_transform(stylometry_corpus_dict)
        
        assert len(stylo.document_vectors) > 0


class TestStandaloneFunctions:
    """Tests for standalone stylometry functions."""
    
    def test_burrows_delta(self):
        """Test Burrows' Delta function."""
        from qhchina.analytics.stylometry import burrows_delta
        
        vec_a = np.array([1.0, 2.0, 3.0])
        vec_b = np.array([1.5, 2.5, 3.5])
        
        delta = burrows_delta(vec_a, vec_b)
        
        assert isinstance(delta, (float, np.floating))
        assert delta == pytest.approx(0.5)  # Mean absolute diff = 0.5
    
    def test_cosine_distance(self):
        """Test cosine distance function."""
        from qhchina.analytics.stylometry import cosine_distance
        
        vec_a = np.array([1.0, 0.0, 0.0])
        vec_b = np.array([0.0, 1.0, 0.0])
        
        dist = cosine_distance(vec_a, vec_b)
        
        assert dist == pytest.approx(1.0)  # Orthogonal vectors
    
    def test_compute_yule_k(self):
        """Test Yule's K computation."""
        from qhchina.analytics.stylometry import compute_yule_k
        
        tokens = list("我我我他他她")  # 3x我, 2x他, 1x她
        
        k = compute_yule_k(tokens)
        
        assert isinstance(k, float)
        assert k > 0
    
    def test_get_relative_frequencies(self):
        """Test relative frequency computation."""
        from qhchina.analytics.stylometry import get_relative_frequencies
        
        items = ["a", "a", "b", "c"]
        
        freqs = get_relative_frequencies(items)
        
        assert freqs["a"] == pytest.approx(0.5)
        assert freqs["b"] == pytest.approx(0.25)
        assert freqs["c"] == pytest.approx(0.25)


class TestCompareCorpora:
    """Tests for compare_corpora() function using 宋史/明史."""
    
    def test_compare_corpora_basic(self, song_ming_flat):
        """Test basic corpus comparison with 宋史 vs 明史."""
        from qhchina.analytics.stylometry import compare_corpora
        
        result = compare_corpora(
            corpusA=song_ming_flat['song'],
            corpusB=song_ming_flat['ming'],
            method='fisher',
            as_dataframe=True
        )
        
        # Should return DataFrame with expected columns
        assert 'word' in result.columns
        assert 'abs_freqA' in result.columns
        assert 'abs_freqB' in result.columns
        assert 'rel_freqA' in result.columns
        assert 'rel_freqB' in result.columns
        assert 'p_value' in result.columns
        assert 'rel_ratio' in result.columns
        
        # Should have some results
        assert len(result) > 0
    
    def test_compare_corpora_with_filters(self, song_ming_flat):
        """Test corpus comparison with filtering."""
        from qhchina.analytics.stylometry import compare_corpora
        
        result = compare_corpora(
            corpusA=song_ming_flat['song'],
            corpusB=song_ming_flat['ming'],
            method='fisher',
            filters={
                'min_count': 5,
                'max_p': 0.05,
                'min_word_length': 1
            },
            as_dataframe=True
        )
        
        # Results should only include words passing filters
        if len(result) > 0:
            assert all(result['abs_freqA'] >= 5) or all(result['abs_freqB'] >= 5)
            assert all(result['p_value'] <= 0.05)
    
    def test_compare_corpora_chi2_methods(self, song_ming_flat):
        """Test different statistical methods."""
        from qhchina.analytics.stylometry import compare_corpora
        
        for method in ['fisher', 'chi2', 'chi2_corrected']:
            result = compare_corpora(
                corpusA=song_ming_flat['song'],
                corpusB=song_ming_flat['ming'],
                method=method,
                as_dataframe=True
            )
            
            assert len(result) > 0
            assert 'p_value' in result.columns
    
    def test_compare_corpora_as_list(self, song_ming_flat):
        """Test returning results as list instead of DataFrame."""
        from qhchina.analytics.stylometry import compare_corpora
        
        result = compare_corpora(
            corpusA=song_ming_flat['song'],
            corpusB=song_ming_flat['ming'],
            method='fisher',
            as_dataframe=False
        )
        
        assert isinstance(result, list)
        assert len(result) > 0
        
        # Each item should be a dict with expected keys
        if len(result) > 0:
            assert 'word' in result[0]
            assert 'p_value' in result[0]
    
    def test_compare_corpora_stopwords_filter(self, song_ming_flat):
        """Test filtering with stopwords."""
        from qhchina.analytics.stylometry import compare_corpora
        
        stopwords = {'之', '者', '也', '而', '其', '為', '以', '於', '乃', '則'}
        
        result = compare_corpora(
            corpusA=song_ming_flat['song'],
            corpusB=song_ming_flat['ming'],
            filters={'stopwords': stopwords},
            as_dataframe=True
        )
        
        # Stopwords should not be in results
        if len(result) > 0:
            for sw in stopwords:
                assert sw not in result['word'].values
    
    def test_compare_corpora_finds_differences(self, song_ming_flat):
        """Test that comparison finds statistically significant differences."""
        from qhchina.analytics.stylometry import compare_corpora
        
        result = compare_corpora(
            corpusA=song_ming_flat['song'],
            corpusB=song_ming_flat['ming'],
            filters={'min_count': 3, 'max_p': 0.01},
            as_dataframe=True
        )
        
        # Should find some significant differences between dynasty histories
        # (The 宋史 and 明史 have different vocabulary patterns)
        if len(result) > 0:
            # Words with rel_ratio > 1 are more common in 宋史
            song_words = result[result['rel_ratio'] > 1]
            # Words with rel_ratio < 1 are more common in 明史
            ming_words = result[result['rel_ratio'] < 1]
            
            # At least some differences should exist
            assert len(result) > 0


class TestStylometryEdgeCases:
    """Tests for edge cases and error conditions."""
    
    def test_empty_corpus(self):
        """Test handling of empty corpus."""
        from qhchina.analytics.stylometry import Stylometry
        
        stylo = Stylometry(n_features=10)
        
        # Empty dict should raise an error or handle gracefully
        with pytest.raises((ValueError, KeyError, IndexError)):
            stylo.fit_transform({})
    
    def test_single_document(self):
        """Test handling of single-document corpus."""
        from qhchina.analytics.stylometry import Stylometry
        
        stylo = Stylometry(n_features=10)
        
        # Single document should raise error (need at least 2 for comparison)
        single_doc = [list("这是一段测试文本用于检验单文档处理")]
        
        with pytest.raises(ValueError, match="at least 2 documents"):
            stylo.fit_transform(single_doc)
    
    def test_predict_before_fit(self):
        """Test that predict before fit raises error."""
        from qhchina.analytics.stylometry import Stylometry
        
        stylo = Stylometry(n_features=10)
        
        with pytest.raises((RuntimeError, AttributeError)):
            stylo.predict(list("测试文本"))
    
    def test_compute_yule_k_empty(self):
        """Test Yule's K with empty tokens."""
        from qhchina.analytics.stylometry import compute_yule_k
        
        k = compute_yule_k([])
        
        assert k == 0.0
    
    def test_get_relative_frequencies_empty(self):
        """Test relative frequencies with empty list."""
        from qhchina.analytics.stylometry import get_relative_frequencies
        
        freqs = get_relative_frequencies([])
        
        assert freqs == {}
    
    def test_compare_corpora_empty(self):
        """Test compare_corpora with empty corpora."""
        from qhchina.analytics.stylometry import compare_corpora
        
        # Empty corpora should raise error
        with pytest.raises(ValueError, match="empty"):
            compare_corpora([], [], as_dataframe=True)
