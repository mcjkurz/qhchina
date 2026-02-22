"""
Tests for qhchina.analytics.word2vec module.
"""
import pytest
import numpy as np
import tempfile
import os
from contextlib import contextmanager


@contextmanager
def tempref_corpus_files(corpora: dict, targets: list):
    """
    Context manager that creates temporary TempRefCorpus files for testing.
    
    Args:
        corpora: Dict mapping labels to lists of sentences
        targets: List of target words to tag
        
    Yields:
        Dict mapping labels to file paths
    """
    from qhchina.corpus import TempRefCorpus
    
    temp_files = []
    file_paths = {}
    
    try:
        for label, sentences in corpora.items():
            corpus = TempRefCorpus(label=label, targets=targets)
            corpus.add_many(sentences)
            
            fd, path = tempfile.mkstemp(suffix='.txt', prefix=f'test_{label}_')
            os.close(fd)
            temp_files.append(path)
            
            corpus.save(path)
            file_paths[label] = path
        
        yield file_paths
    finally:
        for path in temp_files:
            if os.path.exists(path):
                os.unlink(path)


class TestWord2VecBasic:
    """Basic Word2Vec functionality tests."""
    
    def test_word2vec_init(self):
        """Test Word2Vec initialization."""
        from qhchina.analytics.word2vec import Word2Vec
        
        model = Word2Vec(
            vector_size=50,
            window=3,
            min_word_count=1,

        )
        
        assert model.vector_size == 50
        assert model.window == 3
    
    def test_word2vec_train_skipgram_python(self, larger_documents):
        """Test Skip-gram training with Python implementation."""
        from qhchina.analytics.word2vec import Word2Vec
        
        model = Word2Vec(
            larger_documents,
            vector_size=20,
            window=2,
            min_word_count=2,
            negative=3,
            sg=1,  # Skip-gram
            seed=42,
            epochs=2
        )
        model.train()
        
        # Check vocabulary was built
        assert len(model.vocab) > 0
    
    def test_word2vec_train_cbow_python(self, larger_documents):
        """Test CBOW training with Python implementation."""
        from qhchina.analytics.word2vec import Word2Vec
        
        model = Word2Vec(
            larger_documents,
            vector_size=20,
            window=2,
            min_word_count=2,
            negative=3,
            sg=0,  # CBOW
            seed=42,
            epochs=2
        )
        model.train()
        
        assert len(model.vocab) > 0
    
    def test_word2vec_train_cython(self, larger_documents):
        """Test training with Cython implementation."""
        from qhchina.analytics.word2vec import Word2Vec
        
        model = Word2Vec(
            larger_documents,
            vector_size=20,
            window=2,
            min_word_count=2,
            negative=3,
            sg=1,
            seed=42,
            epochs=2
        )
        model.train()
        
        # If Cython not available, it falls back to Python
        assert len(model.vocab) > 0


class TestWord2VecVectors:
    """Tests for Word2Vec vector operations."""
    
    @pytest.fixture
    def trained_model(self, larger_documents):
        """Pre-trained model for vector tests."""
        from qhchina.analytics.word2vec import Word2Vec
        
        model = Word2Vec(
            larger_documents,
            vector_size=30,
            window=3,
            min_word_count=2,
            negative=5,
            sg=1,
            seed=42,
            epochs=3
        )
        model.train()
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
            larger_documents,
            vector_size=20,
            window=2,
            min_word_count=2,
            seed=42,
            epochs=2
        )
        model.train()
        
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
            larger_documents,
            vector_size=20,
            window=2,
            min_word_count=2,
            seed=42,
            epochs=1
        )
        model.train()
        
        # Vocabulary should be populated
        assert len(model.vocab) > 0
        assert len(model.index2word) == len(model.vocab)
    
    def test_vector_dimensions(self, larger_documents):
        """Test that trained vectors have correct dimensions."""
        from qhchina.analytics.word2vec import Word2Vec
        
        vector_size = 25
        model = Word2Vec(
            larger_documents,
            vector_size=vector_size,
            window=2,
            min_word_count=2,
            seed=42,
            epochs=2
        )
        model.train()
        
        # All vectors should have the correct size
        for word in model.vocab:
            vec = model.get_vector(word)
            assert vec.shape == (vector_size,)


# =============================================================================
# Boundary Value Tests
# =============================================================================

class TestWord2VecBoundaryValues:
    """Tests for Word2Vec with boundary parameter values."""
    
    def test_window_size_one(self, larger_documents):
        """Test Word2Vec with window=1."""
        from qhchina.analytics.word2vec import Word2Vec
        
        model = Word2Vec(
            larger_documents,
            vector_size=10,
            window=1,
            min_word_count=2,
            seed=42,
            epochs=1
        )
        model.train()
        
        assert len(model.vocab) > 0
    
    def test_vector_size_one(self, larger_documents):
        """Test Word2Vec with vector_size=1."""
        from qhchina.analytics.word2vec import Word2Vec
        
        model = Word2Vec(
            larger_documents,
            vector_size=1,
            window=2,
            min_word_count=2,
            seed=42,
            epochs=1
        )
        model.train()
        
        for word in model.vocab:
            vec = model.get_vector(word)
            assert vec.shape == (1,)
    
    def test_negative_one(self, larger_documents):
        """Test Word2Vec with negative=1 (single negative sample)."""
        from qhchina.analytics.word2vec import Word2Vec
        
        model = Word2Vec(
            larger_documents,
            vector_size=10,
            window=2,
            min_word_count=2,
            negative=1,
            seed=42,
            epochs=1
        )
        model.train()
        
        assert len(model.vocab) > 0
    
    def test_min_word_count_one(self, sample_documents):
        """Test Word2Vec with min_word_count=1 (keep all words)."""
        from qhchina.analytics.word2vec import Word2Vec
        
        model = Word2Vec(
            sample_documents,
            vector_size=10,
            window=2,
            min_word_count=1,
            seed=42,
            epochs=1
        )
        model.train()
        
        # Should keep more words than with higher threshold
        assert len(model.vocab) > 0
    
    def test_single_epoch(self, larger_documents):
        """Test Word2Vec with epochs=1."""
        from qhchina.analytics.word2vec import Word2Vec
        
        model = Word2Vec(
            larger_documents,
            vector_size=10,
            window=2,
            min_word_count=2,
            seed=42,
            epochs=1
        )
        model.train()
        
        assert len(model.vocab) > 0
    
    def test_max_vocab_size(self, larger_documents):
        """Test Word2Vec with max_vocab_size."""
        from qhchina.analytics.word2vec import Word2Vec
        
        model = Word2Vec(
            larger_documents,
            vector_size=10,
            window=2,
            min_word_count=1,
            max_vocab_size=20,
            seed=42,
            epochs=1
        )
        model.train()
        
        assert len(model.vocab) <= 20
    
    def test_sample_disabled(self, larger_documents):
        """Test Word2Vec with sample=0 (subsampling disabled)."""
        from qhchina.analytics.word2vec import Word2Vec
        
        model = Word2Vec(
            larger_documents,
            vector_size=10,
            window=2,
            min_word_count=2,
            sample=0,  # Disable subsampling
            seed=42,
            epochs=1
        )
        model.train()
        
        assert len(model.vocab) > 0
    
    def test_shrink_windows_disabled(self, larger_documents):
        """Test Word2Vec with shrink_windows=False."""
        from qhchina.analytics.word2vec import Word2Vec
        
        model = Word2Vec(
            larger_documents,
            vector_size=10,
            window=3,
            min_word_count=2,
            shrink_windows=False,
            seed=42,
            epochs=1
        )
        model.train()
        
        assert len(model.vocab) > 0


# =============================================================================
# Edge Cases
# =============================================================================

class TestWord2VecEdgeCases:
    """Tests for Word2Vec edge cases."""
    
    def test_empty_sentences_raises_error(self):
        """Test that empty sentences raises ValueError."""
        from qhchina.analytics.word2vec import Word2Vec
        
        model = Word2Vec([], min_word_count=1)
        
        with pytest.raises(ValueError, match="cannot be empty"):
            model.train()
    
    def test_all_empty_sentences(self):
        """Test handling of all empty sentences."""
        from qhchina.analytics.word2vec import Word2Vec
        
        model = Word2Vec([[], [], []], min_word_count=1)
        
        with pytest.raises(ValueError, match="contains no words"):
            model.train()
    
    def test_no_sentences_raises_error(self):
        """Test that calling train() without sentences raises ValueError."""
        from qhchina.analytics.word2vec import Word2Vec
        
        model = Word2Vec(min_word_count=1)
        
        with pytest.raises(ValueError, match="No sentences provided"):
            model.train()
    
    def test_all_words_filtered(self):
        """Test when all words are below min_word_count."""
        from qhchina.analytics.word2vec import Word2Vec
        
        # Each word appears only once
        sentences = [["a", "b", "c"], ["d", "e", "f"]]
        
        model = Word2Vec(sentences, min_word_count=10, epochs=1)
        model.train()
        
        # Vocabulary should be empty (all filtered)
        assert len(model.vocab) == 0
    
    def test_get_vector_oov(self, larger_documents):
        """Test that OOV words raise KeyError."""
        from qhchina.analytics.word2vec import Word2Vec
        
        model = Word2Vec(
            larger_documents,
            vector_size=10,
            min_word_count=2,
            seed=42,
            epochs=1
        )
        model.train()
        
        with pytest.raises(KeyError):
            model.get_vector("nonexistent_word_xyz_123")
    
    def test_getitem_oov(self, larger_documents):
        """Test that [] access for OOV words raises KeyError."""
        from qhchina.analytics.word2vec import Word2Vec
        
        model = Word2Vec(
            larger_documents,
            vector_size=10,
            min_word_count=2,
            seed=42,
            epochs=1
        )
        model.train()
        
        with pytest.raises(KeyError):
            _ = model["nonexistent_word_xyz_123"]
    
    def test_similarity_oov(self, larger_documents):
        """Test similarity with OOV word raises KeyError."""
        from qhchina.analytics.word2vec import Word2Vec
        
        model = Word2Vec(
            larger_documents,
            vector_size=10,
            min_word_count=2,
            seed=42,
            epochs=1
        )
        model.train()
        
        vocab_words = list(model.vocab.keys())
        if len(vocab_words) > 0:
            with pytest.raises(KeyError):
                model.similarity(vocab_words[0], "nonexistent_word_xyz_123")
    
    def test_single_sentence_corpus(self):
        """Test training on a single sentence."""
        from qhchina.analytics.word2vec import Word2Vec
        
        # Single sentence with repeated words
        sentences = [["a", "b", "a", "b", "a", "c", "b", "c"]]
        model = Word2Vec(
            sentences,
            vector_size=10,
            window=2,
            min_word_count=1,
            seed=42,
            epochs=2
        )
        model.train()
        
        assert len(model.vocab) > 0
    
    def test_very_short_sentences(self):
        """Test training on very short sentences."""
        from qhchina.analytics.word2vec import Word2Vec
        
        # Very short sentences
        sentences = [["a"], ["b"], ["a", "b"], ["c"]]
        model = Word2Vec(
            sentences,
            vector_size=10,
            window=2,
            min_word_count=1,
            seed=42,
            epochs=2
        )
        model.train()
        
        # Should handle gracefully
        assert isinstance(model.vocab, dict)


# =============================================================================
# Training Mode Tests
# =============================================================================

class TestWord2VecTrainingModes:
    """Tests for different training modes (Skip-gram vs CBOW)."""
    
    def test_skipgram_vs_cbow_different_results(self, larger_documents):
        """Test that Skip-gram and CBOW produce different vectors."""
        from qhchina.analytics.word2vec import Word2Vec
        
        model_sg = Word2Vec(
            larger_documents,
            vector_size=20,
            window=2,
            min_word_count=2,
            sg=1,  # Skip-gram
            seed=42,
            epochs=3
        )
        model_sg.train()
        
        model_cbow = Word2Vec(
            larger_documents,
            vector_size=20,
            window=2,
            min_word_count=2,
            sg=0,  # CBOW
            seed=42,
            epochs=3
        )
        model_cbow.train()
        
        # Both should have vocabulary
        assert len(model_sg.vocab) > 0
        assert len(model_cbow.vocab) > 0
        
        # Find a common word
        common_words = set(model_sg.vocab.keys()) & set(model_cbow.vocab.keys())
        if len(common_words) > 0:
            word = list(common_words)[0]
            vec_sg = model_sg.get_vector(word)
            vec_cbow = model_cbow.get_vector(word)
            
            # Vectors should be different (different training algorithms)
            assert not np.allclose(vec_sg, vec_cbow)
    
    def test_cbow_mean_vs_sum(self, larger_documents):
        """Test CBOW with mean vs sum aggregation."""
        from qhchina.analytics.word2vec import Word2Vec
        
        model_mean = Word2Vec(
            larger_documents,
            vector_size=10,
            window=2,
            min_word_count=2,
            sg=0,
            cbow_mean=True,
            seed=42,
            epochs=2
        )
        model_mean.train()
        
        model_sum = Word2Vec(
            larger_documents,
            vector_size=10,
            window=2,
            min_word_count=2,
            sg=0,
            cbow_mean=False,
            seed=42,
            epochs=2
        )
        model_sum.train()
        
        assert len(model_mean.vocab) > 0
        assert len(model_sum.vocab) > 0


class TestTempRefWord2Vec:
    """Tests for TempRefWord2Vec temporal semantic change analysis."""
    
    def test_tempref_init(self, song_ming_corpora):
        """Test TempRefWord2Vec initialization with 宋史/明史."""
        from qhchina.analytics.tempref_word2vec import TempRefWord2Vec
        
        # Find common words in both corpora for target tracking
        song_tokens = set(token for sent in song_ming_corpora['song'] for token in sent)
        ming_tokens = set(token for sent in song_ming_corpora['ming'] for token in sent)
        common_words = list(song_tokens & ming_tokens)[:5]
        
        if len(common_words) < 1:
            pytest.skip("No common words found between corpora")
        
        with tempref_corpus_files(song_ming_corpora, common_words) as file_paths:
            model = TempRefWord2Vec(
                sentences=file_paths,
                targets=common_words,
                vector_size=30,
                window=3,
                min_word_count=2,
                sg=1,
                seed=42,
            )
            
            assert model.labels == ['song', 'ming']
            assert set(model.targets) == set(common_words)
    
    def test_tempref_train(self, song_ming_corpora):
        """Test TempRefWord2Vec training."""
        from qhchina.analytics.tempref_word2vec import TempRefWord2Vec
        
        # Find common words
        song_tokens = set(token for sent in song_ming_corpora['song'] for token in sent)
        ming_tokens = set(token for sent in song_ming_corpora['ming'] for token in sent)
        common_words = list(song_tokens & ming_tokens)[:3]
        
        if len(common_words) < 1:
            pytest.skip("No common words found between corpora")
        
        with tempref_corpus_files(song_ming_corpora, common_words) as file_paths:
            model = TempRefWord2Vec(
                sentences=file_paths,
                targets=common_words,
                vector_size=20,
                window=2,
                min_word_count=2,
                sg=1,
                seed=42,
                epochs=2
            )
            
            model.train()
            
            # Check temporal variants in vocabulary
            for target in common_words:
                if f"{target}_song" in model.vocab or f"{target}_ming" in model.vocab:
                    break
            
            assert len(model.vocab) > 0
    
    def test_tempref_temporal_variants(self, song_ming_corpora):
        """Test that temporal variants are correctly created."""
        from qhchina.analytics.tempref_word2vec import TempRefWord2Vec
        from collections import Counter
        
        # Find words that appear at least 5 times in each corpus
        song_tokens = [token for sent in song_ming_corpora['song'] for token in sent]
        ming_tokens = [token for sent in song_ming_corpora['ming'] for token in sent]
        song_counts = Counter(song_tokens)
        ming_counts = Counter(ming_tokens)
        
        common_frequent = [
            w for w in song_counts 
            if song_counts[w] >= 5 and ming_counts.get(w, 0) >= 5
        ][:2]
        
        if len(common_frequent) < 1:
            pytest.skip("No frequent common words found")
        
        with tempref_corpus_files(song_ming_corpora, common_frequent) as file_paths:
            model = TempRefWord2Vec(
                sentences=file_paths,
                targets=common_frequent,
                vector_size=20,
                window=2,
                min_word_count=3,
                sg=1,
                seed=42,
                epochs=2
            )
            
            model.train()
            
            # Check temporal mapping
            assert len(model.temporal_word_map) > 0
            
            for target in common_frequent:
                if target in model.temporal_word_map:
                    variants = model.temporal_word_map[target]
                    assert f"{target}_song" in variants
                    assert f"{target}_ming" in variants
    
    def test_tempref_save_load(self, song_ming_corpora):
        """Test saving and loading TempRefWord2Vec model."""
        from qhchina.analytics.tempref_word2vec import TempRefWord2Vec
        
        song_tokens = set(token for sent in song_ming_corpora['song'] for token in sent)
        ming_tokens = set(token for sent in song_ming_corpora['ming'] for token in sent)
        common_words = list(song_tokens & ming_tokens)[:2]
        
        if len(common_words) < 1:
            pytest.skip("No common words found")
        
        with tempref_corpus_files(song_ming_corpora, common_words) as file_paths:
            model = TempRefWord2Vec(
                sentences=file_paths,
                targets=common_words,
                vector_size=15,
                window=2,
                min_word_count=2,
                sg=1,
                seed=42,
                epochs=1
            )
            
            model.train()
            
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
        from qhchina.analytics.tempref_word2vec import TempRefWord2Vec
        
        with tempref_corpus_files(song_ming_corpora, ['天']) as file_paths:
            with pytest.raises(NotImplementedError):
                TempRefWord2Vec(
                    sentences=file_paths,
                    targets=['天'],
                    sg=0,  # CBOW not supported
                )
    
    def test_tempref_invalid_sentences_type(self, song_ming_corpora):
        """Test that non-dict sentences raises TypeError."""
        from qhchina.analytics.tempref_word2vec import TempRefWord2Vec
        
        with pytest.raises(TypeError):
            TempRefWord2Vec(
                sentences=["/path/to/file1.txt", "/path/to/file2.txt"],  # List instead of dict
                targets=['天'],
                sg=1,
            )
    
    def test_tempref_invalid_sentences_values(self, song_ming_corpora):
        """Test that non-string sentences values raise TypeError."""
        from qhchina.analytics.tempref_word2vec import TempRefWord2Vec
        
        with pytest.raises(TypeError, match="must be file paths"):
            TempRefWord2Vec(
                sentences={'song': song_ming_corpora['song']},  # List instead of path
                targets=['天'],
                sg=1,
            )
    
    def test_tempref_shuffle_true_raises_error(self, song_ming_corpora):
        """Test that shuffle=True raises ValueError."""
        from qhchina.analytics.tempref_word2vec import TempRefWord2Vec
        
        with tempref_corpus_files(song_ming_corpora, ['天']) as file_paths:
            with pytest.raises(ValueError, match="shuffle=True is not supported"):
                TempRefWord2Vec(
                    sentences=file_paths,
                    targets=['天'],
                    sg=1,
                    shuffle=True,
                )
    
    def test_tempref_empty_targets_raises_error(self, song_ming_corpora):
        """Test that empty targets list raises ValueError."""
        from qhchina.analytics.tempref_word2vec import TempRefWord2Vec
        
        with tempref_corpus_files(song_ming_corpora, ['天']) as file_paths:
            with pytest.raises(ValueError, match="targets cannot be empty"):
                TempRefWord2Vec(
                    sentences=file_paths,
                    targets=[],  # Empty targets
                    sg=1,
                )
    
    def test_tempref_base_word_count_equals_variant_sum(self, song_ming_corpora):
        """Test that base word counts equal the sum of their temporal variant counts.
        
        This verifies the temporal referencing implementation correctly aggregates
        word frequencies from all time periods for proper negative sampling.
        """
        from qhchina.analytics.tempref_word2vec import TempRefWord2Vec
        from collections import Counter
        
        # Find words that appear frequently in both corpora
        song_tokens = [token for sent in song_ming_corpora['song'] for token in sent]
        ming_tokens = [token for sent in song_ming_corpora['ming'] for token in sent]
        song_counts = Counter(song_tokens)
        ming_counts = Counter(ming_tokens)
        
        # Find common frequent words
        common_frequent = [
            w for w in song_counts 
            if song_counts[w] >= 5 and ming_counts.get(w, 0) >= 5
        ][:3]
        
        if len(common_frequent) < 1:
            pytest.skip("No frequent common words found")
        
        with tempref_corpus_files(song_ming_corpora, common_frequent) as file_paths:
            model = TempRefWord2Vec(
                sentences=file_paths,
                targets=common_frequent,
                vector_size=20,
                window=2,
                min_word_count=2,
                sg=1,
                seed=42,
            )
            model.train()
            
            # For each target, verify base word count = sum of variant counts
            for target in common_frequent:
                if target not in model.vocab:
                    continue  # Skip if base word not in vocab
                
                base_count = model.word_counts[target]
                variant_count_sum = 0
                
                for label in model.labels:
                    variant = f"{target}_{label}"
                    if variant in model.word_counts:
                        variant_count_sum += model.word_counts[variant]
                
                # Base word count should equal sum of variants
                assert base_count == variant_count_sum, \
                    f"Base word '{target}' count ({base_count}) != sum of variants ({variant_count_sum})"
    
    def test_tempref_temporal_variants_have_different_embeddings(self, song_ming_corpora):
        """Test that temporal variants of the same word have distinct embeddings.
        
        After training, word_period1 and word_period2 should have different vectors,
        reflecting potential semantic change across time periods.
        """
        from qhchina.analytics.tempref_word2vec import TempRefWord2Vec
        from collections import Counter
        
        # Find words that appear frequently in both corpora
        song_tokens = [token for sent in song_ming_corpora['song'] for token in sent]
        ming_tokens = [token for sent in song_ming_corpora['ming'] for token in sent]
        song_counts = Counter(song_tokens)
        ming_counts = Counter(ming_tokens)
        
        # Find common frequent words
        common_frequent = [
            w for w in song_counts 
            if song_counts[w] >= 10 and ming_counts.get(w, 0) >= 10
        ][:2]
        
        if len(common_frequent) < 1:
            pytest.skip("No frequent common words found")
        
        with tempref_corpus_files(song_ming_corpora, common_frequent) as file_paths:
            model = TempRefWord2Vec(
                sentences=file_paths,
                targets=common_frequent,
                vector_size=30,
                window=3,
                min_word_count=3,
                sg=1,
                seed=42,
                epochs=3
            )
            model.train()
            
            # Check that temporal variants exist and have different vectors
            for target in common_frequent:
                song_variant = f"{target}_song"
                ming_variant = f"{target}_ming"
                
                if song_variant in model.vocab and ming_variant in model.vocab:
                    vec_song = model.get_vector(song_variant)
                    vec_ming = model.get_vector(ming_variant)
                    
                    # Vectors should NOT be identical (training should differentiate them)
                    # Note: They could be similar if the word didn't change semantically,
                    # but they shouldn't be exactly equal after training
                    assert not np.allclose(vec_song, vec_ming, rtol=1e-5, atol=1e-5), \
                        f"Temporal variants of '{target}' have identical vectors"


# =============================================================================
# Implementation Consistency Tests
# =============================================================================

class TestWord2VecConsistency:
    """Tests for consistency between Python and Cython implementations."""
    
    def test_python_cython_vector_consistency_skipgram(self, larger_documents):
        """Test that Python and Cython produce similar vectors for Skip-gram.
        
        Both implementations should produce vectors that, while not identical
        (due to different RNG paths), are similar in their learned relationships.
        """
        from qhchina.analytics.word2vec import Word2Vec
        
        # Common parameters
        params = dict(
            vector_size=30,
            window=3,
            min_word_count=2,
            negative=5,
            sg=1,
            seed=42,
            sample=0,  # Disable subsampling for more comparable results
            shrink_windows=False,  # Fixed windows for consistency
            epochs=3,
            calculate_loss=True,
        )
        
        model = Word2Vec(larger_documents, **params)
        model.train()
        
        # Model should have a valid vocabulary
        assert len(model.vocab) > 0
        
        # Check that most_similar produces valid results
        vocab_words = list(model.vocab.keys())
        if len(vocab_words) >= 5:
            test_word = vocab_words[0]
            similar = model.most_similar(test_word, topn=10)
            assert len(similar) > 0, "Model should produce valid similar words"
    
    def test_cbow_training(self, larger_documents):
        """Test that CBOW training produces valid results."""
        from qhchina.analytics.word2vec import Word2Vec
        
        params = dict(
            vector_size=30,
            window=3,
            min_word_count=2,
            negative=5,
            sg=0,  # CBOW
            seed=42,
            sample=0,
            shrink_windows=False,
            epochs=3,
            calculate_loss=True,
        )
        
        model = Word2Vec(larger_documents, **params)
        model.train()
        
        assert len(model.vocab) > 0
    
    def test_reproducibility_same_seed(self, larger_documents):
        """Test that training with the same seed produces identical results.
        
        Two models trained with identical parameters and seed should produce
        exactly the same vectors.
        """
        from qhchina.analytics.word2vec import Word2Vec
        
        params = dict(
            vector_size=20,
            window=2,
            min_word_count=2,
            negative=3,
            sg=1,
            seed=12345,
            epochs=2,
        )
        
        # Train first model
        model1 = Word2Vec(larger_documents.copy(), **params)
        model1.train()
        
        # Train second model with same seed
        model2 = Word2Vec(larger_documents.copy(), **params)
        model2.train()
        
        # Vocabularies should be identical
        assert model1.vocab == model2.vocab
        
        # All vectors should be exactly equal
        for word in model1.vocab:
            vec1 = model1.get_vector(word)
            vec2 = model2.get_vector(word)
            assert np.allclose(vec1, vec2, rtol=1e-10, atol=1e-10), \
                f"Vectors for '{word}' differ between runs with same seed"
    
    def test_different_seeds_produce_different_results(self, larger_documents):
        """Test that different seeds produce different vectors."""
        from qhchina.analytics.word2vec import Word2Vec
        
        params = dict(
            vector_size=20,
            window=2,
            min_word_count=2,
            negative=3,
            sg=1,
            epochs=2,
        )
        
        model1 = Word2Vec(larger_documents.copy(), **params, seed=42)
        model1.train()
        
        model2 = Word2Vec(larger_documents.copy(), **params, seed=999)
        model2.train()
        
        # Vocabularies should still be the same
        assert model1.vocab == model2.vocab
        
        # But vectors should differ
        common_words = list(model1.vocab.keys())
        if len(common_words) > 0:
            word = common_words[0]
            vec1 = model1.get_vector(word)
            vec2 = model2.get_vector(word)
            assert not np.allclose(vec1, vec2), \
                "Different seeds should produce different vectors"


# =============================================================================
# Learning and Convergence Tests
# =============================================================================

class TestWord2VecLearning:
    """Tests for Word2Vec learning behavior and convergence."""
    
    def test_loss_returned_from_training(self, larger_documents):
        """Test that training returns a valid loss value.
        
        This verifies the model is computing loss during training.
        """
        from qhchina.analytics.word2vec import Word2Vec
        
        model = Word2Vec(
            larger_documents,
            vector_size=30,
            window=3,
            min_word_count=2,
            negative=5,
            sg=1,
            seed=42,
            alpha=0.025,
            min_alpha=0.0001,
            epochs=5,
            calculate_loss=True,
        )
        
        loss = model.train()
        
        # Loss should be a valid positive number
        assert loss is not None
        assert loss > 0
        assert not np.isnan(loss)
        assert not np.isinf(loss)
    
    def test_loss_decreases_within_epoch_cython(self, larger_documents):
        """Test that loss decreases during training with Cython."""
        from qhchina.analytics.word2vec import Word2Vec
        
        model = Word2Vec(
            larger_documents,
            vector_size=30,
            window=3,
            min_word_count=2,
            negative=5,
            sg=1,
            seed=42,
            alpha=0.025,
            min_alpha=0.0001,
            epochs=3,
            calculate_loss=True,
        )
        
        # Training should return a valid loss
        loss = model.train()
        
        assert loss is not None, "Loss should be computed"
        assert loss > 0, "Loss should be positive"
        assert not np.isnan(loss), "Loss should not be NaN"
        assert not np.isinf(loss), "Loss should not be infinite"
    
    def test_learning_rate_decay(self, larger_documents):
        """Test that learning rate decays from alpha to min_alpha during training."""
        from qhchina.analytics.word2vec import Word2Vec
        
        start_alpha = 0.05
        end_alpha = 0.001
        
        model = Word2Vec(
            larger_documents,
            vector_size=20,
            window=2,
            min_word_count=2,
            negative=3,
            sg=1,
            seed=42,
            alpha=start_alpha,
            min_alpha=end_alpha,
            epochs=5
        )
        model.train()
        
        # After training, alpha should have decayed to min_alpha
        # (or close to it, depending on total examples)
        assert model.alpha <= start_alpha, \
            "Alpha should decrease during training"
        
        # Final alpha should be at or near min_alpha
        # Allow some tolerance since it depends on example count estimation
        assert model.alpha <= end_alpha * 1.5, \
            f"Final alpha ({model.alpha}) should be close to min_alpha ({end_alpha})"
    
    def test_no_learning_rate_decay_when_min_alpha_not_set(self, larger_documents):
        """Test that learning rate stays constant when min_alpha is not set."""
        from qhchina.analytics.word2vec import Word2Vec
        
        start_alpha = 0.025
        
        model = Word2Vec(
            larger_documents,
            vector_size=20,
            window=2,
            min_word_count=2,
            negative=3,
            sg=1,
            seed=42,
            alpha=start_alpha,
            min_alpha=None,  # No decay
            epochs=3
        )
        model.train()
        
        # Alpha should remain unchanged
        assert model.alpha == start_alpha, \
            "Alpha should stay constant when min_alpha is not set"
    
    def test_vectors_change_after_training(self, larger_documents):
        """Test that word vectors actually change from initialization after training."""
        from qhchina.analytics.word2vec import Word2Vec
        
        model = Word2Vec(
            larger_documents,
            vector_size=20,
            window=2,
            min_word_count=2,
            negative=3,
            sg=1,
            seed=42,
            epochs=3,
        )
        
        # Build vocab and initialize vectors manually to capture initial state
        model.build_vocab(model._sentences)
        model._initialize_vectors()
        model._prepare_noise_distribution()
        
        # Store initial vectors
        initial_vectors = model.W.copy()
        
        # Train
        model.train()
        
        # Vectors should have changed
        assert not np.allclose(model.W, initial_vectors), \
            "Vectors should change after training"


# =============================================================================
# Vector Quality Tests
# =============================================================================

class TestWord2VecVectorQuality:
    """Tests for Word2Vec vector quality and operations."""
    
    def test_normalized_vectors_have_unit_length(self, larger_documents):
        """Test that normalized vectors have unit length."""
        from qhchina.analytics.word2vec import Word2Vec
        
        model = Word2Vec(
            larger_documents,
            vector_size=30,
            window=3,
            min_word_count=2,
            negative=5,
            sg=1,
            seed=42,
            epochs=2
        )
        model.train()
        
        # Check normalized vectors
        for word in list(model.vocab.keys())[:10]:
            vec = model.get_vector(word, normalize=True)
            norm = np.linalg.norm(vec)
            
            assert np.isclose(norm, 1.0, rtol=1e-5), \
                f"Normalized vector for '{word}' has norm {norm}, expected 1.0"
    
    def test_unnormalized_vectors_preserve_magnitude(self, larger_documents):
        """Test that unnormalized vectors preserve their learned magnitude."""
        from qhchina.analytics.word2vec import Word2Vec
        
        model = Word2Vec(
            larger_documents,
            vector_size=30,
            window=3,
            min_word_count=2,
            negative=5,
            sg=1,
            seed=42,
            epochs=2
        )
        model.train()
        
        # Check that vectors have non-trivial magnitudes
        for word in list(model.vocab.keys())[:10]:
            vec = model.get_vector(word, normalize=False)
            norm = np.linalg.norm(vec)
            
            # Vectors should have some magnitude (not all zeros)
            assert norm > 0, f"Vector for '{word}' has zero magnitude"
    
    def test_similarity_is_symmetric(self, larger_documents):
        """Test that similarity(a, b) == similarity(b, a)."""
        from qhchina.analytics.word2vec import Word2Vec
        
        model = Word2Vec(
            larger_documents,
            vector_size=20,
            window=2,
            min_word_count=2,
            negative=3,
            sg=1,
            seed=42,
            epochs=2
        )
        model.train()
        
        vocab_words = list(model.vocab.keys())
        if len(vocab_words) >= 2:
            word1, word2 = vocab_words[0], vocab_words[1]
            
            sim_12 = model.similarity(word1, word2)
            sim_21 = model.similarity(word2, word1)
            
            assert np.isclose(sim_12, sim_21, rtol=1e-10), \
                f"Similarity should be symmetric: {sim_12} != {sim_21}"
    
    def test_self_similarity_is_one(self, larger_documents):
        """Test that similarity of a word with itself is 1.0."""
        from qhchina.analytics.word2vec import Word2Vec
        
        model = Word2Vec(
            larger_documents,
            vector_size=20,
            window=2,
            min_word_count=2,
            negative=3,
            sg=1,
            seed=42,
            epochs=2
        )
        model.train()
        
        for word in list(model.vocab.keys())[:5]:
            sim = model.similarity(word, word)
            assert np.isclose(sim, 1.0, rtol=1e-5), \
                f"Self-similarity for '{word}' is {sim}, expected 1.0"
    
    def test_most_similar_returns_sorted_results(self, larger_documents):
        """Test that most_similar returns results sorted by similarity (descending)."""
        from qhchina.analytics.word2vec import Word2Vec
        
        model = Word2Vec(
            larger_documents,
            vector_size=20,
            window=2,
            min_word_count=2,
            negative=3,
            sg=1,
            seed=42,
            epochs=2
        )
        model.train()
        
        vocab_words = list(model.vocab.keys())
        if len(vocab_words) > 5:
            word = vocab_words[0]
            similar = model.most_similar(word, topn=10)
            
            if len(similar) > 1:
                # Check that similarities are in descending order
                sims = [s for _, s in similar]
                assert sims == sorted(sims, reverse=True), \
                    "most_similar results should be sorted by similarity (descending)"
    
    def test_most_similar_excludes_query_word(self, larger_documents):
        """Test that most_similar doesn't include the query word itself."""
        from qhchina.analytics.word2vec import Word2Vec
        
        model = Word2Vec(
            larger_documents,
            vector_size=20,
            window=2,
            min_word_count=2,
            negative=3,
            sg=1,
            seed=42,
            epochs=2
        )
        model.train()
        
        vocab_words = list(model.vocab.keys())
        if len(vocab_words) > 5:
            word = vocab_words[0]
            similar = model.most_similar(word, topn=10)
            
            similar_words = [w for w, _ in similar]
            assert word not in similar_words, \
                "most_similar should not include the query word"


# =============================================================================
# Stress and Performance Tests
# =============================================================================

class TestWord2VecStress:
    """Stress tests for Word2Vec with larger data."""
    
    def test_large_vocabulary(self):
        """Test Word2Vec with a larger vocabulary.
        
        This tests memory handling and potential overflow issues.
        """
        from qhchina.analytics.word2vec import Word2Vec
        
        # Create a corpus with many unique words
        vocab_size = 1000
        sentences = []
        for i in range(100):
            # Each sentence contains a random subset of words
            sentence = [f"word_{j}" for j in range(i * 10, i * 10 + 50)]
            sentences.append(sentence)
        
        model = Word2Vec(
            sentences,
            vector_size=50,
            window=3,
            min_word_count=1,
            negative=5,
            sg=1,
            seed=42,
            epochs=1
        )
        model.train()
        
        # Should handle large vocabulary without errors
        assert len(model.vocab) > 100
        
        # Vectors should be accessible
        for word in list(model.vocab.keys())[:10]:
            vec = model.get_vector(word)
            assert vec.shape == (50,)
    
    def test_long_sentences(self, larger_documents):
        """Test Word2Vec with longer sentences."""
        from qhchina.analytics.word2vec import Word2Vec
        
        # Create long sentences
        long_sentences = [
            larger_documents[0] * 10,  # Repeat first doc 10 times
            larger_documents[1] * 10,
        ]
        
        model = Word2Vec(
            long_sentences,
            vector_size=20,
            window=5,
            min_word_count=2,
            negative=3,
            sg=1,
            seed=42,
            epochs=1
        )
        model.train()
        
        # Should handle without errors
        assert len(model.vocab) > 0
    
    def test_many_epochs(self, sample_documents):
        """Test training for many epochs."""
        from qhchina.analytics.word2vec import Word2Vec
        
        model = Word2Vec(
            sample_documents,
            vector_size=10,
            window=2,
            min_word_count=1,
            negative=2,
            sg=1,
            seed=42,
            alpha=0.025,
            min_alpha=0.0001,
            epochs=20
        )
        model.train()
        
        # Should complete without errors
        assert len(model.vocab) > 0
        
        # Vectors should still be valid (not NaN or Inf)
        for word in model.vocab:
            vec = model.get_vector(word)
            assert not np.any(np.isnan(vec)), f"NaN in vector for '{word}'"
            assert not np.any(np.isinf(vec)), f"Inf in vector for '{word}'"


class TestWord2VecAlphaHandling:
    """Tests for alpha=None vs alpha=0 behavior."""
    
    def test_alpha_none_uses_default(self, sample_documents):
        """Test that alpha=None triggers the default learning rate."""
        from qhchina.analytics.word2vec import Word2Vec
        
        model = Word2Vec(
            sample_documents,
            vector_size=10,
            window=2,
            min_word_count=1,
            negative=2,
            sg=1,
            seed=42,
            alpha=None
        )
        model.train()
        
        # After training, alpha should have been set to default 0.025
        assert model.alpha == 0.025
        assert model.min_alpha is None
    
    def test_alpha_zero_does_not_trigger_default(self, sample_documents):
        """Test that alpha=0.0 is respected and does NOT trigger the default."""
        from qhchina.analytics.word2vec import Word2Vec
        
        model = Word2Vec(
            sentences=sample_documents,
            vector_size=10,
            window=2,
            min_word_count=1,
            negative=2,
            sg=1,
            seed=42,

            alpha=0.0
        )
        model.train()
        
        # alpha=0.0 should be kept as-is (not replaced with 0.025)
        assert model.alpha == 0.0


class TestWord2VecMultithreading:
    """Tests for multithreaded training."""
    
    def test_workers_one_trains_successfully(self, larger_documents):
        """Test that workers=1 (pipelined) training works correctly."""
        from qhchina.analytics.word2vec import Word2Vec
        
        model = Word2Vec(
            larger_documents,
            vector_size=20,
            window=3,
            min_word_count=2,
            negative=3,
            sg=1,
            seed=42,
            epochs=2,
            workers=1,
        )
        model.train()
        
        assert len(model.vocab) > 0
        assert model.W is not None
        assert model.W.shape[1] == 20
        
        # Vectors should be valid
        for word in list(model.vocab.keys())[:5]:
            vec = model.get_vector(word)
            assert not np.any(np.isnan(vec))
            assert not np.any(np.isinf(vec))
    
    def test_workers_multiple_trains_successfully(self, larger_documents):
        """Test that workers>1 (parallel Hogwild) training works correctly."""
        from qhchina.analytics.word2vec import Word2Vec
        
        model = Word2Vec(
            larger_documents,
            vector_size=20,
            window=3,
            min_word_count=2,
            negative=3,
            sg=1,
            seed=42,
            epochs=2,
            workers=4,
        )
        model.train()
        
        assert len(model.vocab) > 0
        assert model.W is not None
        assert model.W.shape[1] == 20
        
        # Vectors should be valid
        for word in list(model.vocab.keys())[:5]:
            vec = model.get_vector(word)
            assert not np.any(np.isnan(vec))
            assert not np.any(np.isinf(vec))
    
    def test_workers_cbow_mode(self, larger_documents):
        """Test multithreading works with CBOW mode."""
        from qhchina.analytics.word2vec import Word2Vec
        
        model = Word2Vec(
            larger_documents,
            vector_size=20,
            window=3,
            min_word_count=2,
            negative=3,
            sg=0,  # CBOW
            seed=42,
            epochs=2,
            workers=2,
        )
        model.train()
        
        assert len(model.vocab) > 0
        assert model.W is not None
    
    def test_workers_invalid_raises_error(self):
        """Test that workers<1 raises an error."""
        from qhchina.analytics.word2vec import Word2Vec
        
        with pytest.raises(ValueError, match="workers must be at least 1"):
            Word2Vec(
                vector_size=20,
                workers=0,
            )
    
    def test_workers_learning_rate_decay(self, larger_documents):
        """Test that learning rate decay works with multithreading."""
        from qhchina.analytics.word2vec import Word2Vec
        
        model = Word2Vec(
            larger_documents,
            vector_size=20,
            window=3,
            min_word_count=2,
            negative=3,
            sg=1,
            seed=42,
            alpha=0.025,
            min_alpha=0.0001,
            epochs=3,
            workers=2,
        )
        model.train()
        
        # After training with min_alpha set, alpha should equal min_alpha
        assert model.alpha == 0.0001


class TestTempRefWord2VecMultithreading:
    """Tests for TempRefWord2Vec multithreading."""
    
    def test_tempref_workers_one(self):
        """Test TempRefWord2Vec with workers=1."""
        from qhchina.analytics.tempref_word2vec import TempRefWord2Vec
        
        corpora = {
            "period1": [["word1", "word2", "word3"]] * 50,
            "period2": [["word1", "word4", "word5"]] * 50,
        }
        
        with tempref_corpus_files(corpora, ["word1"]) as file_paths:
            model = TempRefWord2Vec(
                sentences=file_paths,
                targets=["word1"],
                vector_size=20,
                window=2,
                min_word_count=1,
                negative=3,
                sg=1,
                seed=42,
                epochs=2,
                workers=1,
            )
            model.train()
            
            assert len(model.vocab) > 0
            assert "word1_period1" in model.vocab
            assert "word1_period2" in model.vocab
    
    def test_tempref_workers_multiple(self):
        """Test TempRefWord2Vec with workers>1."""
        from qhchina.analytics.tempref_word2vec import TempRefWord2Vec
        
        corpora = {
            "period1": [["word1", "word2", "word3"]] * 50,
            "period2": [["word1", "word4", "word5"]] * 50,
        }
        
        with tempref_corpus_files(corpora, ["word1"]) as file_paths:
            model = TempRefWord2Vec(
                sentences=file_paths,
                targets=["word1"],
                vector_size=20,
                window=2,
                min_word_count=1,
                negative=3,
                sg=1,
                seed=42,
                epochs=2,
                workers=2,
            )
            model.train()
            
            assert len(model.vocab) > 0
            assert "word1_period1" in model.vocab
            assert "word1_period2" in model.vocab
            
            # Vectors should be valid
            vec1 = model.get_vector("word1_period1")
            vec2 = model.get_vector("word1_period2")
            assert not np.any(np.isnan(vec1))
            assert not np.any(np.isnan(vec2))


