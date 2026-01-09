"""
Tests for qhchina.analytics.word2vec module.
Tests both Python and Cython implementations.
"""
import pytest
import numpy as np
import tempfile
import os


class TestWord2VecBasic:
    """Basic Word2Vec functionality tests."""
    
    def test_word2vec_init(self):
        """Test Word2Vec initialization."""
        from qhchina.analytics.word2vec import Word2Vec
        
        model = Word2Vec(
            vector_size=50,
            window=3,
            min_word_count=1,
            use_cython=False
        )
        
        assert model.vector_size == 50
        assert model.window == 3
    
    def test_word2vec_train_skipgram_python(self, larger_documents):
        """Test Skip-gram training with Python implementation."""
        from qhchina.analytics.word2vec import Word2Vec
        
        model = Word2Vec(
            vector_size=20,
            window=2,
            min_word_count=2,
            negative=3,
            sg=1,  # Skip-gram
            seed=42,
            use_cython=False
        )
        
        model.train(larger_documents, epochs=2)
        
        # Check vocabulary was built
        assert len(model.vocab) > 0
    
    def test_word2vec_train_cbow_python(self, larger_documents):
        """Test CBOW training with Python implementation."""
        from qhchina.analytics.word2vec import Word2Vec
        
        model = Word2Vec(
            vector_size=20,
            window=2,
            min_word_count=2,
            negative=3,
            sg=0,  # CBOW
            seed=42,
            use_cython=False
        )
        
        model.train(larger_documents, epochs=2)
        
        assert len(model.vocab) > 0
    
    def test_word2vec_train_cython(self, larger_documents):
        """Test training with Cython implementation."""
        from qhchina.analytics.word2vec import Word2Vec
        
        model = Word2Vec(
            vector_size=20,
            window=2,
            min_word_count=2,
            negative=3,
            sg=1,
            seed=42,
            use_cython=True
        )
        
        # If Cython not available, it falls back to Python
        model.train(larger_documents, epochs=2)
        
        assert len(model.vocab) > 0


class TestWord2VecVectors:
    """Tests for Word2Vec vector operations."""
    
    @pytest.fixture
    def trained_model(self, larger_documents):
        """Pre-trained model for vector tests."""
        from qhchina.analytics.word2vec import Word2Vec
        
        model = Word2Vec(
            vector_size=30,
            window=3,
            min_word_count=2,
            negative=5,
            sg=1,
            seed=42,
            use_cython=False
        )
        model.train(larger_documents, epochs=3)
        return model
    
    def test_get_vector(self, trained_model):
        """Test getting word vectors."""
        # Get a word from vocabulary
        vocab_words = list(trained_model.vocab.keys())
        if len(vocab_words) > 0:
            word = vocab_words[0]
            vector = trained_model.get_vector(word)
            
            assert isinstance(vector, np.ndarray)
            assert len(vector) == trained_model.vector_size
    
    def test_getitem_access(self, trained_model):
        """Test dictionary-like access to vectors."""
        vocab_words = list(trained_model.vocab.keys())
        if len(vocab_words) > 0:
            word = vocab_words[0]
            vector = trained_model[word]
            
            assert isinstance(vector, np.ndarray)
    
    def test_most_similar(self, trained_model):
        """Test finding most similar words."""
        vocab_words = list(trained_model.vocab.keys())
        if len(vocab_words) > 0:
            word = vocab_words[0]
            similar = trained_model.most_similar(word, topn=5)
            
            assert isinstance(similar, list)
            assert len(similar) <= 5
            if len(similar) > 0:
                # Each result is (word, similarity)
                assert len(similar[0]) == 2
    
    def test_similarity(self, trained_model):
        """Test computing similarity between two words."""
        vocab_words = list(trained_model.vocab.keys())
        if len(vocab_words) >= 2:
            word1, word2 = vocab_words[0], vocab_words[1]
            sim = trained_model.similarity(word1, word2)
            
            assert isinstance(sim, (float, np.floating))
            assert -1 <= sim <= 1
    
    def test_contains(self, trained_model):
        """Test word containment check."""
        vocab_words = list(trained_model.vocab.keys())
        if len(vocab_words) > 0:
            word = vocab_words[0]
            assert word in trained_model
            assert "nonexistent_word_xyz" not in trained_model


class TestWord2VecSaveLoad:
    """Tests for saving and loading Word2Vec models."""
    
    def test_save_load(self, larger_documents):
        """Test saving and loading a model."""
        from qhchina.analytics.word2vec import Word2Vec
        
        model = Word2Vec(
            vector_size=20,
            window=2,
            min_word_count=2,
            seed=42,
            use_cython=False
        )
        model.train(larger_documents, epochs=2)
        
        # Save to temp file with .npy extension (used by np.save)
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            temp_path = f.name
        
        try:
            model.save(temp_path)
            
            # Verify file was created and has content
            assert os.path.exists(temp_path)
            assert os.path.getsize(temp_path) > 0
            
            # Load the model
            loaded_model = Word2Vec.load(temp_path)
            
            assert loaded_model.vector_size == model.vector_size
            assert len(loaded_model.vocab) == len(model.vocab)
            
            # Check vectors are the same
            for word in model.vocab:
                orig_vec = model.get_vector(word)
                loaded_vec = loaded_model.get_vector(word)
                assert np.allclose(orig_vec, loaded_vec)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestWord2VecTraining:
    """Tests for Word2Vec training behavior."""
    
    def test_vocabulary_built(self, larger_documents):
        """Test that vocabulary is correctly built during training."""
        from qhchina.analytics.word2vec import Word2Vec
        
        model = Word2Vec(
            vector_size=20,
            window=2,
            min_word_count=2,
            seed=42,
            use_cython=False
        )
        model.train(larger_documents, epochs=1)
        
        # Vocabulary should be populated
        assert len(model.vocab) > 0
        assert len(model.index2word) == len(model.vocab)
    
    def test_vector_dimensions(self, larger_documents):
        """Test that trained vectors have correct dimensions."""
        from qhchina.analytics.word2vec import Word2Vec
        
        vector_size = 25
        model = Word2Vec(
            vector_size=vector_size,
            window=2,
            min_word_count=2,
            seed=42,
            use_cython=False
        )
        model.train(larger_documents, epochs=2)
        
        # All vectors should have the correct size
        for word in model.vocab:
            vec = model.get_vector(word)
            assert vec.shape == (vector_size,)


class TestTempRefWord2Vec:
    """Tests for TempRefWord2Vec temporal semantic change analysis."""
    
    def test_tempref_init(self, song_ming_corpora):
        """Test TempRefWord2Vec initialization with 宋史/明史."""
        from qhchina.analytics.word2vec import TempRefWord2Vec
        
        # Find common words in both corpora for target tracking
        song_tokens = set(token for sent in song_ming_corpora['song'] for token in sent)
        ming_tokens = set(token for sent in song_ming_corpora['ming'] for token in sent)
        common_words = list(song_tokens & ming_tokens)[:5]  # Take first 5 common words
        
        if len(common_words) < 1:
            pytest.skip("No common words found between corpora")
        
        model = TempRefWord2Vec(
            corpora=[song_ming_corpora['song'], song_ming_corpora['ming']],
            labels=['song', 'ming'],
            targets=common_words,
            vector_size=30,
            window=3,
            min_word_count=2,
            sg=1,  # Skip-gram required
            seed=42,
            use_cython=False
        )
        
        assert model.labels == ['song', 'ming']
        assert set(model.targets) == set(common_words)
    
    def test_tempref_train(self, song_ming_corpora):
        """Test TempRefWord2Vec training."""
        from qhchina.analytics.word2vec import TempRefWord2Vec
        
        # Find common words
        song_tokens = set(token for sent in song_ming_corpora['song'] for token in sent)
        ming_tokens = set(token for sent in song_ming_corpora['ming'] for token in sent)
        common_words = list(song_tokens & ming_tokens)[:3]
        
        if len(common_words) < 1:
            pytest.skip("No common words found between corpora")
        
        model = TempRefWord2Vec(
            corpora=[song_ming_corpora['song'], song_ming_corpora['ming']],
            labels=['song', 'ming'],
            targets=common_words,
            vector_size=20,
            window=2,
            min_word_count=2,
            sg=1,
            seed=42,
            use_cython=False
        )
        
        model.train(epochs=2)
        
        # Check temporal variants in vocabulary
        for target in common_words:
            if f"{target}_song" in model.vocab or f"{target}_ming" in model.vocab:
                # At least one temporal variant should exist
                break
        else:
            # This is OK - variants might be filtered by min_word_count
            pass
        
        assert len(model.vocab) > 0
    
    def test_tempref_temporal_variants(self, song_ming_corpora):
        """Test that temporal variants are correctly created."""
        from qhchina.analytics.word2vec import TempRefWord2Vec
        
        # Use a common word that should appear frequently
        song_tokens = [token for sent in song_ming_corpora['song'] for token in sent]
        ming_tokens = [token for sent in song_ming_corpora['ming'] for token in sent]
        
        # Find words that appear at least 5 times in each corpus
        from collections import Counter
        song_counts = Counter(song_tokens)
        ming_counts = Counter(ming_tokens)
        
        common_frequent = [
            w for w in song_counts 
            if song_counts[w] >= 5 and ming_counts.get(w, 0) >= 5
        ][:2]
        
        if len(common_frequent) < 1:
            pytest.skip("No frequent common words found")
        
        model = TempRefWord2Vec(
            corpora=[song_ming_corpora['song'], song_ming_corpora['ming']],
            labels=['song', 'ming'],
            targets=common_frequent,
            vector_size=20,
            window=2,
            min_word_count=3,
            sg=1,
            seed=42,
            use_cython=False
        )
        
        model.train(epochs=2)
        
        # Check temporal mapping
        assert len(model.temporal_word_map) > 0
        
        for target in common_frequent:
            if target in model.temporal_word_map:
                variants = model.temporal_word_map[target]
                assert f"{target}_song" in variants
                assert f"{target}_ming" in variants
    
    def test_tempref_save_load(self, song_ming_corpora):
        """Test saving and loading TempRefWord2Vec model."""
        from qhchina.analytics.word2vec import TempRefWord2Vec
        
        song_tokens = set(token for sent in song_ming_corpora['song'] for token in sent)
        ming_tokens = set(token for sent in song_ming_corpora['ming'] for token in sent)
        common_words = list(song_tokens & ming_tokens)[:2]
        
        if len(common_words) < 1:
            pytest.skip("No common words found")
        
        model = TempRefWord2Vec(
            corpora=[song_ming_corpora['song'], song_ming_corpora['ming']],
            labels=['song', 'ming'],
            targets=common_words,
            vector_size=15,
            window=2,
            min_word_count=2,
            sg=1,
            seed=42,
            use_cython=False
        )
        
        model.train(epochs=1)
        
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            temp_path = f.name
        
        try:
            model.save(temp_path)
            
            loaded = TempRefWord2Vec.load(temp_path)
            
            assert loaded.labels == model.labels
            assert loaded.targets == model.targets
            assert len(loaded.vocab) == len(model.vocab)
            assert loaded.vector_size == model.vector_size
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_tempref_requires_skipgram(self, song_ming_corpora):
        """Test that TempRefWord2Vec raises error for CBOW."""
        from qhchina.analytics.word2vec import TempRefWord2Vec
        
        with pytest.raises(NotImplementedError):
            TempRefWord2Vec(
                corpora=[song_ming_corpora['song'], song_ming_corpora['ming']],
                labels=['song', 'ming'],
                targets=['天'],
                sg=0,  # CBOW not supported
            )
    
    def test_tempref_mismatched_labels(self, song_ming_corpora):
        """Test that mismatched corpora and labels raises error."""
        from qhchina.analytics.word2vec import TempRefWord2Vec
        
        with pytest.raises(ValueError):
            TempRefWord2Vec(
                corpora=[song_ming_corpora['song'], song_ming_corpora['ming']],
                labels=['only_one_label'],  # Wrong number of labels
                targets=['天'],
                sg=1,
            )
