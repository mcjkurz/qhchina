"""
Tests for qhchina.analytics.topicmodels module.
Tests both Python and Cython implementations.
"""
import pytest
import numpy as np
import tempfile
import os


class TestLDABasic:
    """Basic LDA functionality tests."""
    
    def test_lda_init(self):
        """Test LDA initialization."""
        from qhchina.analytics.topicmodels import LDAGibbsSampler
        
        lda = LDAGibbsSampler(
            n_topics=5,
            iterations=10,
            random_state=42
        )
        
        assert lda.n_topics == 5
        assert lda.iterations == 10
    
    def test_lda_init_validation(self):
        """Test that invalid parameters raise errors."""
        from qhchina.analytics.topicmodels import LDAGibbsSampler
        
        with pytest.raises(ValueError):
            LDAGibbsSampler(n_topics=-1)
        
        with pytest.raises(ValueError):
            LDAGibbsSampler(iterations=0)
    
    def test_lda_fit_python(self, sample_documents):
        """Test LDA fitting with Python implementation."""
        from qhchina.analytics.topicmodels import LDAGibbsSampler
        
        lda = LDAGibbsSampler(
            n_topics=3,
            iterations=10,
            random_state=42,
            use_cython=False,
            min_word_count=1
        )
        
        lda.fit(sample_documents)
        
        assert lda.n_wt is not None
        assert lda.n_dt is not None
        assert lda.vocabulary is not None
    
    def test_lda_fit_cython(self, sample_documents):
        """Test LDA fitting with Cython implementation."""
        from qhchina.analytics.topicmodels import LDAGibbsSampler
        
        lda = LDAGibbsSampler(
            n_topics=3,
            iterations=10,
            random_state=42,
            use_cython=True,
            min_word_count=1
        )
        
        # Falls back to Python if Cython not available
        lda.fit(sample_documents)
        
        assert lda.n_wt is not None


class TestLDATopics:
    """Tests for LDA topic extraction."""
    
    @pytest.fixture
    def fitted_lda(self, larger_documents):
        """Pre-fitted LDA model."""
        from qhchina.analytics.topicmodels import LDAGibbsSampler
        
        lda = LDAGibbsSampler(
            n_topics=5,
            iterations=20,
            random_state=42,
            use_cython=False,
            min_word_count=2
        )
        lda.fit(larger_documents)
        return lda
    
    def test_get_topic_words(self, fitted_lda):
        """Test getting top words for a topic."""
        words = fitted_lda.get_topic_words(topic_id=0, n_words=10)
        
        assert isinstance(words, list)
        assert len(words) <= 10
        if len(words) > 0:
            # Each item is (word, probability)
            assert len(words[0]) == 2
            assert isinstance(words[0][0], str)
            assert isinstance(words[0][1], (float, np.floating))
    
    def test_get_document_topics(self, fitted_lda):
        """Test getting topic distribution for a document."""
        # get_document_topics returns list of (topic_id, probability) tuples
        topics = fitted_lda.get_document_topics(doc_id=0)
        
        assert isinstance(topics, list)
        assert len(topics) == fitted_lda.n_topics
        # Each item is (topic_id, probability)
        assert all(len(t) == 2 for t in topics)
        # Probabilities should sum to ~1
        probs = [p for _, p in topics]
        assert np.isclose(sum(probs), 1.0, rtol=0.1)


class TestLDAInference:
    """Tests for LDA inference on new documents."""
    
    @pytest.fixture
    def fitted_lda(self, larger_documents):
        """Pre-fitted LDA model."""
        from qhchina.analytics.topicmodels import LDAGibbsSampler
        
        lda = LDAGibbsSampler(
            n_topics=5,
            iterations=20,
            random_state=42,
            use_cython=False,
            min_word_count=2
        )
        lda.fit(larger_documents)
        return lda
    
    def test_inference(self, fitted_lda, sample_tokenized):
        """Test inference on a new document."""
        topic_dist = fitted_lda.inference(sample_tokenized, inference_iterations=10)
        
        assert isinstance(topic_dist, np.ndarray)
        assert len(topic_dist) == fitted_lda.n_topics
        assert np.isclose(topic_dist.sum(), 1.0, rtol=0.1)


class TestLDASaveLoad:
    """Tests for saving and loading LDA models."""
    
    def test_save_load(self, sample_documents):
        """Test saving and loading a model."""
        from qhchina.analytics.topicmodels import LDAGibbsSampler
        
        lda = LDAGibbsSampler(
            n_topics=3,
            iterations=10,
            random_state=42,
            use_cython=False,
            min_word_count=1
        )
        lda.fit(sample_documents)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            temp_path = f.name
        
        try:
            lda.save(temp_path)
            
            # Load the model
            loaded_lda = LDAGibbsSampler.load(temp_path)
            
            assert loaded_lda.n_topics == lda.n_topics
            assert loaded_lda.vocabulary == lda.vocabulary
            assert np.array_equal(loaded_lda.n_wt, lda.n_wt)
        finally:
            os.unlink(temp_path)


class TestLDAReproducibility:
    """Tests for reproducibility with random seeds."""
    
    def test_reproducibility_with_seed(self, sample_documents):
        """Test that same seed gives same results."""
        from qhchina.analytics.topicmodels import LDAGibbsSampler
        
        lda1 = LDAGibbsSampler(
            n_topics=3,
            iterations=15,
            random_state=42,
            use_cython=False,
            min_word_count=1
        )
        lda1.fit(sample_documents)
        
        lda2 = LDAGibbsSampler(
            n_topics=3,
            iterations=15,
            random_state=42,
            use_cython=False,
            min_word_count=1
        )
        lda2.fit(sample_documents)
        
        # Same seed should give same word-topic counts
        assert np.array_equal(lda1.n_wt, lda2.n_wt)


class TestLDAParameters:
    """Tests for LDA hyperparameters."""
    
    def test_custom_alpha_beta(self, sample_documents):
        """Test custom alpha and beta values."""
        from qhchina.analytics.topicmodels import LDAGibbsSampler
        
        lda = LDAGibbsSampler(
            n_topics=3,
            alpha=0.5,
            beta=0.1,
            iterations=10,
            random_state=42,
            use_cython=False,
            min_word_count=1
        )
        lda.fit(sample_documents)
        
        assert lda.n_wt is not None
    
    def test_vocabulary_filtering(self, larger_documents):
        """Test vocabulary filtering with min_word_count."""
        from qhchina.analytics.topicmodels import LDAGibbsSampler
        
        lda_loose = LDAGibbsSampler(
            n_topics=3,
            iterations=5,
            random_state=42,
            use_cython=False,
            min_word_count=1
        )
        lda_loose.fit(larger_documents)
        
        lda_strict = LDAGibbsSampler(
            n_topics=3,
            iterations=5,
            random_state=42,
            use_cython=False,
            min_word_count=3
        )
        lda_strict.fit(larger_documents)
        
        # Stricter filtering should result in smaller vocabulary
        assert len(lda_strict.vocabulary) <= len(lda_loose.vocabulary)
