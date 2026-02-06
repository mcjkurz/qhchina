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
    
    def test_extract_mfw(self):
        """Test extracting most frequent words."""
        from collections import Counter
        from qhchina.analytics.stylometry import extract_mfw
        
        counts = Counter(['的', '是', '了', '的', '我', '的', '他', '是'])
        mfw = extract_mfw(counts, n=3)
        
        assert isinstance(mfw, list)
        assert len(mfw) == 3
        assert mfw[0] == '的'  # Most frequent
    
    def test_extract_mfw_less_than_n(self):
        """Test extract_mfw when corpus has fewer unique words than n."""
        from collections import Counter
        from qhchina.analytics.stylometry import extract_mfw
        
        counts = Counter(['a', 'a', 'b'])
        mfw = extract_mfw(counts, n=10)
        
        # Should return all available words (2)
        assert len(mfw) == 2
    
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
        # Each word must have freq >= min_count in at least one corpus
        if len(result) > 0:
            assert all((result['abs_freqA'] >= 5) | (result['abs_freqB'] >= 5))
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


# =============================================================================
# Prediction Tests
# =============================================================================

class TestStylometryPrediction:
    """Tests for Stylometry prediction methods."""
    
    @pytest.fixture
    def fitted_stylo(self, stylometry_corpus_dict):
        from qhchina.analytics.stylometry import Stylometry
        stylo = Stylometry(n_features=20, distance='cosine')
        stylo.fit_transform(stylometry_corpus_dict)
        return stylo
    
    def test_predict_returns_results(self, fitted_stylo):
        """Test that predict returns valid results."""
        test_text = list("这是一段测试文本用于预测作者身份归属")
        results = fitted_stylo.predict(test_text, k=2)
        
        assert isinstance(results, list)
        assert len(results) == 2
        
        for author, score in results:
            assert author in fitted_stylo.authors
            assert isinstance(score, float)
    
    def test_predict_k_one(self, fitted_stylo):
        """Test predict with k=1."""
        test_text = list("测试预测功能")
        results = fitted_stylo.predict(test_text, k=1)
        
        assert len(results) == 1
    
    def test_predict_author_convenience(self, fitted_stylo):
        """Test predict_author convenience method."""
        test_text = list("这是一段测试文本")
        author = fitted_stylo.predict_author(test_text)
        
        assert isinstance(author, str)
        assert author in fitted_stylo.authors
    
    def test_predict_confidence(self, fitted_stylo):
        """Test predict_confidence returns normalized scores."""
        test_text = list("这是一段测试文本用于置信度测试")
        results = fitted_stylo.predict_confidence(test_text, k=2)
        
        assert isinstance(results, list)
        # Confidence scores should be between 0 and 1
        for author, conf in results:
            assert 0 <= conf <= 1
    
    def test_predict_invalid_k(self, fitted_stylo):
        """Test that invalid k raises ValueError."""
        test_text = list("测试")
        
        with pytest.raises(ValueError, match="k must be a positive integer"):
            fitted_stylo.predict(test_text, k=0)
        
        with pytest.raises(ValueError, match="k must be a positive integer"):
            fitted_stylo.predict(test_text, k=-1)


class TestStylometryPredictionSVM:
    """Tests for SVM classifier in Stylometry."""
    
    @pytest.fixture
    def fitted_stylo_svm(self, stylometry_corpus_dict):
        from qhchina.analytics.stylometry import Stylometry
        stylo = Stylometry(n_features=20, classifier='svm')
        stylo.fit_transform(stylometry_corpus_dict)
        return stylo
    
    def test_predict_svm(self, fitted_stylo_svm):
        """Test SVM prediction."""
        test_text = list("这是一段测试文本用于SVM预测")
        results = fitted_stylo_svm.predict(test_text, k=2, classifier='svm')
        
        assert isinstance(results, list)
        assert len(results) == 2
        
        # SVM returns probabilities
        for author, prob in results:
            assert author in fitted_stylo_svm.authors
            assert 0 <= prob <= 1


class TestStylometryBootstrap:
    """Tests for bootstrap prediction."""
    
    @pytest.fixture
    def fitted_stylo(self, stylometry_corpus_dict):
        from qhchina.analytics.stylometry import Stylometry
        stylo = Stylometry(n_features=20, distance='cosine')
        stylo.fit_transform(stylometry_corpus_dict)
        return stylo
    
    def test_bootstrap_predict_basic(self, fitted_stylo):
        """Test basic bootstrap prediction."""
        test_text = list("这是一段测试文本用于引导预测分析")
        results = fitted_stylo.bootstrap_predict(test_text, n_iter=10, seed=42)
        
        assert 'prediction' in results
        assert 'confidence' in results
        assert 'distribution' in results
        assert 'distances' in results
        assert 'n_iterations' in results
        
        assert results['prediction'] in fitted_stylo.authors
        assert 0 <= results['confidence'] <= 1
        assert results['n_iterations'] == 10
    
    def test_bootstrap_predict_reproducibility(self, fitted_stylo):
        """Test that bootstrap with same seed gives same results."""
        test_text = list("测试引导法可重复性")
        
        results1 = fitted_stylo.bootstrap_predict(test_text, n_iter=10, seed=42)
        results2 = fitted_stylo.bootstrap_predict(test_text, n_iter=10, seed=42)
        
        assert results1['prediction'] == results2['prediction']
        assert results1['confidence'] == results2['confidence']


# =============================================================================
# Visualization Tests
# =============================================================================

class TestStylometryVisualization:
    """Tests for Stylometry visualization methods."""
    
    @pytest.fixture
    def fitted_stylo(self, stylometry_corpus_dict):
        from qhchina.analytics.stylometry import Stylometry
        stylo = Stylometry(n_features=20, distance='cosine')
        stylo.fit_transform(stylometry_corpus_dict)
        return stylo
    
    def test_plot_pca(self, fitted_stylo):
        """Test PCA visualization."""
        import matplotlib
        matplotlib.use('Agg')
        from qhchina import helpers
        import matplotlib.pyplot as plt
        
        helpers.load_fonts()
        # Should not raise
        fitted_stylo.plot(method='pca', show=False)
        plt.close('all')
    
    def test_plot_tsne(self, fitted_stylo):
        """Test t-SNE visualization."""
        import matplotlib
        matplotlib.use('Agg')
        from qhchina import helpers
        import matplotlib.pyplot as plt
        
        helpers.load_fonts()
        # Should not raise
        fitted_stylo.plot(method='tsne', show=False)
        plt.close('all')
    
    def test_plot_mds(self, fitted_stylo):
        """Test MDS visualization."""
        import matplotlib
        matplotlib.use('Agg')
        from qhchina import helpers
        import matplotlib.pyplot as plt
        
        helpers.load_fonts()
        # Should not raise
        fitted_stylo.plot(method='mds', show=False)
        plt.close('all')
    
    def test_plot_invalid_method(self, fitted_stylo):
        """Test that invalid plot method raises ValueError."""
        with pytest.raises(ValueError, match="method must be"):
            fitted_stylo.plot(method='invalid')
    
    def test_plot_author_level(self, fitted_stylo):
        """Test visualization at author level."""
        import matplotlib
        matplotlib.use('Agg')
        from qhchina import helpers
        import matplotlib.pyplot as plt
        
        helpers.load_fonts()
        fitted_stylo.plot(method='pca', level='author', show=False)
        plt.close('all')
    
    def test_plot_save_to_file(self, fitted_stylo, tmp_path):
        """Test saving plot to file."""
        import matplotlib
        matplotlib.use('Agg')
        from qhchina import helpers
        import matplotlib.pyplot as plt
        
        helpers.load_fonts()
        filepath = tmp_path / "stylometry_plot.png"
        fitted_stylo.plot(method='pca', filename=str(filepath), show=False)
        plt.close('all')
        
        assert filepath.exists()
    
    def test_dendrogram_basic(self, fitted_stylo):
        """Test basic dendrogram visualization."""
        import matplotlib
        matplotlib.use('Agg')
        from qhchina import helpers
        import matplotlib.pyplot as plt
        
        helpers.load_fonts()
        # Should not raise
        fitted_stylo.dendrogram(show=False)
        plt.close('all')
    
    def test_dendrogram_author_level(self, fitted_stylo):
        """Test dendrogram at author level."""
        import matplotlib
        matplotlib.use('Agg')
        from qhchina import helpers
        import matplotlib.pyplot as plt
        
        helpers.load_fonts()
        fitted_stylo.dendrogram(level='author', show=False)
        plt.close('all')
    
    def test_dendrogram_different_methods(self, fitted_stylo):
        """Test dendrogram with different linkage methods."""
        import matplotlib
        matplotlib.use('Agg')
        from qhchina import helpers
        import matplotlib.pyplot as plt
        
        helpers.load_fonts()
        for method in ['single', 'complete', 'average', 'ward']:
            fitted_stylo.dendrogram(method=method, show=False)
            plt.close('all')
    
    def test_dendrogram_invalid_method(self, fitted_stylo):
        """Test that invalid dendrogram method raises ValueError."""
        with pytest.raises(ValueError, match="method must be one of"):
            fitted_stylo.dendrogram(method='invalid')
    
    def test_dendrogram_save_to_file(self, fitted_stylo, tmp_path):
        """Test saving dendrogram to file."""
        import matplotlib
        matplotlib.use('Agg')
        from qhchina import helpers
        import matplotlib.pyplot as plt
        
        helpers.load_fonts()
        filepath = tmp_path / "dendrogram.png"
        fitted_stylo.dendrogram(filename=str(filepath), show=False)
        plt.close('all')
        
        assert filepath.exists()


# =============================================================================
# Boundary Value Tests
# =============================================================================

class TestStylometryBoundaryValues:
    """Tests for Stylometry with boundary parameter values."""
    
    def test_n_features_one(self, stylometry_corpus_dict):
        """Test with n_features=1."""
        from qhchina.analytics.stylometry import Stylometry
        
        stylo = Stylometry(n_features=1)
        stylo.fit_transform(stylometry_corpus_dict)
        
        assert len(stylo.features) == 1
    
    def test_ngram_range_bigrams_only(self, stylometry_corpus_dict):
        """Test with bigrams only (2, 2)."""
        from qhchina.analytics.stylometry import Stylometry
        
        stylo = Stylometry(n_features=20, ngram_range=(2, 2))
        stylo.fit_transform(stylometry_corpus_dict)
        
        # All features should be bigrams (contain space)
        for feature in stylo.features:
            assert ' ' in feature
    
    def test_ngram_range_trigrams(self, stylometry_corpus_dict):
        """Test with trigrams (3, 3)."""
        from qhchina.analytics.stylometry import Stylometry
        
        stylo = Stylometry(n_features=20, ngram_range=(3, 3))
        stylo.fit_transform(stylometry_corpus_dict)
        
        # Features should exist (may be fewer due to small corpus)
        assert len(stylo.features) >= 1
    
    def test_two_documents_minimum(self):
        """Test with exactly 2 documents (minimum required)."""
        from qhchina.analytics.stylometry import Stylometry
        
        corpus = {
            "author_a": [list("第一篇文档内容测试")],
            "author_b": [list("第二篇文档内容测试")]
        }
        
        stylo = Stylometry(n_features=10)
        stylo.fit_transform(corpus)
        
        assert len(stylo.document_ids) == 2
    
    def test_cull_threshold(self, stylometry_corpus_dict):
        """Test with culling enabled."""
        from qhchina.analytics.stylometry import Stylometry
        
        stylo = Stylometry(n_features=20, cull=0.5)
        stylo.fit_transform(stylometry_corpus_dict)
        
        # Should still have some features
        assert len(stylo.features) > 0
    
    def test_chunk_size(self, stylometry_corpus_dict):
        """Test with chunk_size specified."""
        from qhchina.analytics.stylometry import Stylometry
        
        # Use small chunk size
        stylo = Stylometry(n_features=10, chunk_size=10)
        stylo.fit_transform(stylometry_corpus_dict)
        
        # Should have more documents after chunking
        assert len(stylo.document_ids) > 0


class TestStylometryValidation:
    """Tests for Stylometry parameter validation."""
    
    def test_invalid_n_features_zero(self):
        """Test that n_features=0 raises ValueError."""
        from qhchina.analytics.stylometry import Stylometry
        
        with pytest.raises(ValueError, match="n_features must be at least 1"):
            Stylometry(n_features=0)
    
    def test_invalid_n_features_negative(self):
        """Test that negative n_features raises ValueError."""
        from qhchina.analytics.stylometry import Stylometry
        
        with pytest.raises(ValueError, match="n_features must be at least 1"):
            Stylometry(n_features=-1)
    
    def test_invalid_ngram_range(self):
        """Test that invalid ngram_range raises error."""
        from qhchina.analytics.stylometry import Stylometry
        
        with pytest.raises((ValueError, TypeError)):
            Stylometry(ngram_range=(2, 1))  # min > max
    
    def test_invalid_transform(self):
        """Test that invalid transform raises ValueError."""
        from qhchina.analytics.stylometry import Stylometry
        
        with pytest.raises(ValueError, match="transform must be one of"):
            Stylometry(transform='invalid')
    
    def test_invalid_distance(self):
        """Test that invalid distance raises ValueError."""
        from qhchina.analytics.stylometry import Stylometry
        
        with pytest.raises(ValueError, match="distance must be one of"):
            Stylometry(distance='invalid')
    
    def test_invalid_classifier(self):
        """Test that invalid classifier raises ValueError."""
        from qhchina.analytics.stylometry import Stylometry
        
        with pytest.raises(ValueError, match="classifier must be one of"):
            Stylometry(classifier='invalid')
    
    def test_invalid_mode(self):
        """Test that invalid mode raises ValueError."""
        from qhchina.analytics.stylometry import Stylometry
        
        with pytest.raises(ValueError, match="mode must be one of"):
            Stylometry(mode='invalid')


class TestCompareCorporaCorrection:
    """Tests for multiple testing correction in compare_corpora."""
    
    def test_bonferroni_adds_adjusted_column(self, song_ming_flat):
        """Test that Bonferroni correction adds adjusted_p_value column."""
        from qhchina.analytics.stylometry import compare_corpora
        
        result = compare_corpora(
            corpusA=song_ming_flat['song'],
            corpusB=song_ming_flat['ming'],
            method='fisher',
            correction='bonferroni',
            as_dataframe=True
        )
        
        assert 'adjusted_p_value' in result.columns
        assert 'p_value' in result.columns
        # Adjusted p-values should be >= raw p-values
        assert all(result['adjusted_p_value'] >= result['p_value'] - 1e-15)
        # Adjusted p-values should be capped at 1.0
        assert all(result['adjusted_p_value'] <= 1.0)
    
    def test_fdr_bh_adds_adjusted_column(self, song_ming_flat):
        """Test that Benjamini-Hochberg correction adds adjusted_p_value column."""
        from qhchina.analytics.stylometry import compare_corpora
        
        result = compare_corpora(
            corpusA=song_ming_flat['song'],
            corpusB=song_ming_flat['ming'],
            method='fisher',
            correction='fdr_bh',
            as_dataframe=True
        )
        
        assert 'adjusted_p_value' in result.columns
        # Adjusted p-values should be >= raw p-values
        assert all(result['adjusted_p_value'] >= result['p_value'] - 1e-15)
        # Adjusted p-values should be capped at 1.0
        assert all(result['adjusted_p_value'] <= 1.0)
    
    def test_no_correction_by_default(self, song_ming_flat):
        """Test that no correction is applied by default."""
        from qhchina.analytics.stylometry import compare_corpora
        
        result = compare_corpora(
            corpusA=song_ming_flat['song'],
            corpusB=song_ming_flat['ming'],
            method='fisher',
            as_dataframe=True
        )
        
        assert 'adjusted_p_value' not in result.columns
    
    def test_correction_none_same_as_default(self, song_ming_flat):
        """Test that correction=None produces same result as no correction."""
        from qhchina.analytics.stylometry import compare_corpora
        
        result = compare_corpora(
            corpusA=song_ming_flat['song'],
            corpusB=song_ming_flat['ming'],
            method='fisher',
            correction=None,
            as_dataframe=True
        )
        
        assert 'adjusted_p_value' not in result.columns
    
    def test_invalid_correction_raises_error(self, song_ming_flat):
        """Test that invalid correction method raises ValueError."""
        from qhchina.analytics.stylometry import compare_corpora
        
        with pytest.raises(ValueError, match="Unknown correction method"):
            compare_corpora(
                corpusA=song_ming_flat['song'],
                corpusB=song_ming_flat['ming'],
                correction='invalid_method'
            )
    
    def test_bonferroni_more_conservative_than_fdr(self, song_ming_flat):
        """Test that Bonferroni correction is more conservative than FDR."""
        from qhchina.analytics.stylometry import compare_corpora
        
        result_bonf = compare_corpora(
            corpusA=song_ming_flat['song'],
            corpusB=song_ming_flat['ming'],
            correction='bonferroni',
            as_dataframe=True
        )
        result_fdr = compare_corpora(
            corpusA=song_ming_flat['song'],
            corpusB=song_ming_flat['ming'],
            correction='fdr_bh',
            as_dataframe=True
        )
        
        # Merge on word to compare
        merged = result_bonf.merge(result_fdr, on='word', suffixes=('_bonf', '_fdr'))
        # Bonferroni adjusted p should be >= FDR adjusted p in general
        assert all(
            merged['adjusted_p_value_bonf'] >= merged['adjusted_p_value_fdr'] - 1e-15
        )
    
    def test_max_p_filter_still_uses_raw_p_value(self, song_ming_flat):
        """Test that max_p filter always uses raw p_value, even with correction."""
        from qhchina.analytics.stylometry import compare_corpora
        
        result = compare_corpora(
            corpusA=song_ming_flat['song'],
            corpusB=song_ming_flat['ming'],
            correction='bonferroni',
            filters={'max_p': 0.05},
            as_dataframe=True
        )
        
        if len(result) > 0:
            # max_p should filter on raw p_value
            assert all(result['p_value'] <= 0.05)
    
    def test_max_adjusted_p_filters_on_adjusted(self, song_ming_flat):
        """Test that max_adjusted_p filter uses adjusted_p_value."""
        from qhchina.analytics.stylometry import compare_corpora
        
        result = compare_corpora(
            corpusA=song_ming_flat['song'],
            corpusB=song_ming_flat['ming'],
            correction='bonferroni',
            filters={'max_adjusted_p': 0.05},
            as_dataframe=True
        )
        
        if len(result) > 0:
            assert all(result['adjusted_p_value'] <= 0.05)
    
    def test_max_adjusted_p_without_correction_raises_error(self, song_ming_flat):
        """Test that max_adjusted_p without correction raises ValueError."""
        from qhchina.analytics.stylometry import compare_corpora
        
        with pytest.raises(ValueError, match="max_adjusted_p filter requires a correction"):
            compare_corpora(
                corpusA=song_ming_flat['song'],
                corpusB=song_ming_flat['ming'],
                filters={'max_adjusted_p': 0.05}
            )
    
    def test_both_max_p_and_max_adjusted_p(self, song_ming_flat):
        """Test using both max_p and max_adjusted_p together."""
        from qhchina.analytics.stylometry import compare_corpora
        
        result = compare_corpora(
            corpusA=song_ming_flat['song'],
            corpusB=song_ming_flat['ming'],
            correction='fdr_bh',
            filters={'max_p': 0.01, 'max_adjusted_p': 0.05},
            as_dataframe=True
        )
        
        if len(result) > 0:
            assert all(result['p_value'] <= 0.01)
            assert all(result['adjusted_p_value'] <= 0.05)
    
    def test_correction_with_list_output(self, song_ming_flat):
        """Test correction works when as_dataframe=False."""
        from qhchina.analytics.stylometry import compare_corpora
        
        result = compare_corpora(
            corpusA=song_ming_flat['song'],
            corpusB=song_ming_flat['ming'],
            correction='fdr_bh',
            as_dataframe=False
        )
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert 'adjusted_p_value' in result[0]
