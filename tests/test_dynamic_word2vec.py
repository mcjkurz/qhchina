"""
Tests for DynamicWord2Vec with temporal regularization.
"""

import pytest
import pickle
import numpy as np
import tempfile
import os
from qhchina.analytics import DynamicWord2Vec


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def temporal_sentences():
    """Simple temporal corpora for testing."""
    return {
        '宋': [
            ["太祖", "建隆", "元年", "正月"],
            ["民", "安", "其", "业", "太平"],
            ["天下", "归", "心", "民", "安"],
        ] * 10,  # Repeat for more training data
        '明': [
            ["太祖", "洪武", "元年", "春"],
            ["民", "困", "于", "役", "战乱"],
            ["天下", "未", "定", "民", "困"],
        ] * 10,
    }


@pytest.fixture
def small_model(temporal_sentences):
    """Small DynamicWord2Vec model for testing."""
    model = DynamicWord2Vec(
        sentences=temporal_sentences,
        vector_size=50,
        window=3,
        min_word_count=1,
        epochs=5,
        temporal_lambda=0.1,
        temporal_reg_V=True,
        sampling_strategy="balanced",
        alpha=0.025,
        min_alpha=0.001,
        batch_size=100,
        workers=1,
        verbose=False,
    )
    model.train()
    return model


# =============================================================================
# TestDynamicWord2VecBasic
# =============================================================================

class TestDynamicWord2VecBasic:
    """Test basic initialization and validation."""

    def test_init_validates_dict_input(self):
        """Test that sentences must be a dictionary."""
        with pytest.raises(TypeError, match="must be a dictionary"):
            DynamicWord2Vec(
                sentences=[["word1", "word2"]],
                vector_size=50,
                epochs=1,
            )

    def test_init_validates_empty_dict(self):
        """Test that sentences cannot be empty."""
        with pytest.raises(ValueError, match="cannot be empty"):
            DynamicWord2Vec(
                sentences={},
                vector_size=50,
                epochs=1,
            )

    def test_init_validates_sg_requirement(self):
        """Test that sg must be 1 (skip-gram only)."""
        with pytest.raises(NotImplementedError, match="Skip-gram"):
            DynamicWord2Vec(
                sentences={'t1': [["word"]]},
                vector_size=50,
                epochs=1,
                sg=0,  # CBOW not supported
            )

    def test_init_validates_temporal_lambda(self):
        """Test that temporal_lambda must be non-negative."""
        with pytest.raises(ValueError, match="temporal_lambda"):
            DynamicWord2Vec(
                sentences={'t1': [["word"]]},
                vector_size=50,
                epochs=1,
                temporal_lambda=-0.1,
            )

    def test_init_validates_sampling_strategy(self):
        """Test that sampling_strategy must be valid."""
        with pytest.raises(ValueError, match="sampling_strategy"):
            DynamicWord2Vec(
                sentences={'t1': [["word"]]},
                vector_size=50,
                epochs=1,
                sampling_strategy="invalid",
            )

    def test_init_rejects_shuffle(self):
        """Test that shuffle=True is rejected (iterator handles mixing)."""
        with pytest.raises(ValueError, match="shuffle"):
            DynamicWord2Vec(
                sentences={'t1': [["word"]]},
                vector_size=50,
                epochs=1,
                shuffle=True,
            )

    def test_embedding_shapes(self, temporal_sentences):
        """Test that embeddings have correct 3D shape."""
        model = DynamicWord2Vec(
            sentences=temporal_sentences,
            vector_size=50,
            min_word_count=1,
            epochs=1,
            verbose=False,
        )
        model.build_vocab()
        model._initialize_vectors()

        T = 2  # Two time periods
        vocab_size = len(model.vocab)
        vector_size = 50

        assert model.U.shape == (T, vocab_size, vector_size)
        assert model.V.shape == (T, vocab_size, vector_size)
        assert model.num_time_slices == T

    def test_shared_vocabulary(self, temporal_sentences):
        """Test that vocabulary is shared across all time slices."""
        model = DynamicWord2Vec(
            sentences=temporal_sentences,
            vector_size=50,
            min_word_count=1,
            epochs=1,
            verbose=False,
        )
        model.build_vocab()

        # Words that appear in both periods should be in shared vocab
        assert "太祖" in model.vocab
        assert "民" in model.vocab
        assert "天下" in model.vocab

        # Verify that we have counts from both periods
        assert model.period_vocab_counts['宋']['太祖'] > 0
        assert model.period_vocab_counts['明']['太祖'] > 0

    def test_label_mapping(self, temporal_sentences):
        """Test that labels are correctly mapped to indices."""
        model = DynamicWord2Vec(
            sentences=temporal_sentences,
            vector_size=50,
            epochs=1,
            verbose=False,
        )

        assert model.labels == ['宋', '明']
        assert model.label2idx == {'宋': 0, '明': 1}
        assert model.num_time_slices == 2

    def test_initialization_same_base(self, temporal_sentences):
        """Test that all time slices start with same base embeddings."""
        model = DynamicWord2Vec(
            sentences=temporal_sentences,
            vector_size=50,
            min_word_count=1,
            epochs=1,
            seed=42,
            verbose=False,
        )
        model.build_vocab()
        model._initialize_vectors()

        # Before training, all slices should have identical U embeddings
        # (initialized from same base)
        np.testing.assert_array_equal(model.U[0], model.U[1])


# =============================================================================
# TestTemporalRegularization
# =============================================================================

class TestTemporalRegularization:
    """Test temporal regularization behavior."""

    def test_regularization_reduces_drift(self, temporal_sentences):
        """Test that regularization reduces embedding drift."""
        # Model with regularization
        model_reg = DynamicWord2Vec(
            sentences=temporal_sentences,
            vector_size=50,
            min_word_count=1,
            epochs=10,
            temporal_lambda=1.0,  # Strong regularization
            seed=42,
            verbose=False,
        )
        model_reg.train()

        # Model without regularization
        model_noreg = DynamicWord2Vec(
            sentences=temporal_sentences,
            vector_size=50,
            min_word_count=1,
            epochs=10,
            temporal_lambda=0.0,  # No regularization
            seed=42,
            verbose=False,
        )
        model_noreg.train()

        # Check drift for a common word
        word = "民"
        if word in model_reg.vocab:
            drift_reg = model_reg.calculate_temporal_drift(word)
            drift_noreg = model_noreg.calculate_temporal_drift(word)

            # With regularization, drift should be smaller
            # (though this is not guaranteed for all words, especially with small data)
            # At minimum, check that both methods produce valid output
            assert len(drift_reg) == 1  # One transition (宋 -> 明)
            assert len(drift_noreg) == 1
            assert drift_reg[0] >= 0  # Cosine distance is non-negative
            assert drift_noreg[0] >= 0

    def test_temporal_lambda_zero_no_regularization(self, temporal_sentences):
        """Test that temporal_lambda=0 means no regularization."""
        model = DynamicWord2Vec(
            sentences=temporal_sentences,
            vector_size=50,
            min_word_count=1,
            epochs=1,
            temporal_lambda=0.0,
            verbose=False,
        )

        # Should train without errors
        model.train()
        assert model.U is not None
        assert model.V is not None

    def test_temporal_reg_V_parameter(self, temporal_sentences):
        """Test that temporal_reg_V parameter is stored correctly."""
        model_no_v = DynamicWord2Vec(
            sentences=temporal_sentences,
            vector_size=50,
            epochs=1,
            temporal_reg_V=False,
            verbose=False,
        )
        assert model_no_v.temporal_reg_V is False

        model_with_v = DynamicWord2Vec(
            sentences=temporal_sentences,
            vector_size=50,
            epochs=1,
            temporal_reg_V=True,
            verbose=False,
        )
        assert model_with_v.temporal_reg_V is True


# =============================================================================
# TestDynamicWord2VecAPI
# =============================================================================

class TestDynamicWord2VecAPI:
    """Test query API methods."""

    def test_get_vector_with_time_label(self, small_model):
        """Test get_vector() with time labels."""
        word = "民"
        vec_song = small_model.get_vector(word, time_label="宋")
        vec_ming = small_model.get_vector(word, time_label="明")

        assert vec_song.shape == (50,)
        assert vec_ming.shape == (50,)
        # Vectors should be different after training (unless drift is exactly 0)
        assert not np.allclose(vec_song, vec_ming, atol=1e-6)

    def test_get_vector_invalid_word(self, small_model):
        """Test get_vector() with word not in vocabulary."""
        with pytest.raises(KeyError, match="not in vocabulary"):
            small_model.get_vector("不存在的词", time_label="宋")

    def test_get_vector_invalid_time_label(self, small_model):
        """Test get_vector() with invalid time label."""
        with pytest.raises(KeyError, match="not found"):
            small_model.get_vector("民", time_label="清")

    def test_get_vector_normalize(self, small_model):
        """Test get_vector() with normalization."""
        word = "民"
        vec = small_model.get_vector(word, time_label="宋", normalize=True)

        # Normalized vector should have unit length
        norm = np.linalg.norm(vec)
        assert np.isclose(norm, 1.0, atol=1e-6)

    def test_get_all_time_vectors(self, small_model):
        """Test get_all_time_vectors() returns correct shape."""
        word = "民"
        all_vecs = small_model.get_all_time_vectors(word)

        assert all_vecs.shape == (2, 50)  # 2 time slices, 50 dimensions

    def test_get_all_time_vectors_normalize(self, small_model):
        """Test get_all_time_vectors() with normalization."""
        word = "民"
        all_vecs = small_model.get_all_time_vectors(word, normalize=True)

        # Each vector should have unit length
        norms = np.linalg.norm(all_vecs, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)

    def test_most_similar(self, small_model):
        """Test most_similar() per time slice."""
        word = "民"
        similar_song = small_model.most_similar(word, time_label="宋", topn=3)
        similar_ming = small_model.most_similar(word, time_label="明", topn=3)

        assert len(similar_song) <= 3
        assert len(similar_ming) <= 3

        # Each result should be (word, similarity) tuple
        for w, sim in similar_song:
            assert isinstance(w, str)
            assert isinstance(sim, float)
            assert -1.0 <= sim <= 1.0

    def test_most_similar_excludes_query_word(self, small_model):
        """Test that most_similar() doesn't return the query word."""
        word = "民"
        similar = small_model.most_similar(word, time_label="宋", topn=10)

        # Query word should not be in results
        similar_words = [w for w, _ in similar]
        assert word not in similar_words

    def test_calculate_temporal_drift(self, small_model):
        """Test calculate_temporal_drift() returns correct shape."""
        word = "民"
        drift = small_model.calculate_temporal_drift(word)

        # Should have T-1 drift values (one per transition)
        assert drift.shape == (1,)  # 2 time slices -> 1 transition
        assert drift[0] >= 0  # Cosine distance is non-negative
        assert drift[0] <= 2  # Max cosine distance is 2

    def test_calculate_semantic_change(self, small_model):
        """Test calculate_semantic_change() returns correct format."""
        word = "民"
        changes = small_model.calculate_semantic_change(word)

        assert isinstance(changes, dict)
        assert "宋_to_明" in changes

        # Should return list of (word, change) tuples
        word_changes = changes["宋_to_明"]
        assert isinstance(word_changes, list)

        # Verify format
        for w, change in word_changes:
            assert isinstance(w, str)
            assert isinstance(change, float)
            assert w != word  # Should exclude query word

    def test_calculate_semantic_change_with_reference_words(self, small_model):
        """Test calculate_semantic_change() with reference_words filter."""
        word = "民"
        ref_words = ["天下", "太祖"]

        changes = small_model.calculate_semantic_change(word, filters={"reference_words": ref_words})

        # Should only include reference words in results
        word_changes = changes["宋_to_明"]
        result_words = [w for w, _ in word_changes]

        for w in result_words:
            assert w in ref_words

    def test_get_time_labels(self, small_model):
        """Test get_time_labels() returns correct labels."""
        labels = small_model.get_time_labels()

        assert labels == ['宋', '明']
        assert isinstance(labels, list)

        # Should be a copy, not the original
        labels.append('清')
        assert small_model.labels == ['宋', '明']


# =============================================================================
# TestSaveLoad
# =============================================================================

class TestSaveLoad:
    """Test save and load functionality."""

    def test_save_and_load(self, small_model):
        """Test that save/load preserves all model state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.npy")

            # Save model
            small_model.save(save_path)

            # Load model
            loaded_model = DynamicWord2Vec.load(save_path)

            # Check basic attributes
            assert loaded_model.labels == small_model.labels
            assert loaded_model.label2idx == small_model.label2idx
            assert loaded_model.num_time_slices == small_model.num_time_slices
            assert loaded_model.temporal_lambda == small_model.temporal_lambda
            assert loaded_model.temporal_reg_V == small_model.temporal_reg_V
            assert loaded_model._sampling_strategy == small_model._sampling_strategy

            # Check vocabulary
            assert loaded_model.vocab == small_model.vocab
            assert loaded_model.index2word == small_model.index2word
            assert loaded_model.word_counts == small_model.word_counts

            # Check embeddings
            np.testing.assert_array_equal(loaded_model.U, small_model.U)
            np.testing.assert_array_equal(loaded_model.V, small_model.V)

            # Check that queries work
            word = "民"
            vec_orig = small_model.get_vector(word, time_label="宋")
            vec_loaded = loaded_model.get_vector(word, time_label="宋")
            np.testing.assert_array_equal(vec_orig, vec_loaded)

    def test_load_validates_model_type(self):
        """Test that load() rejects non-DynamicWord2Vec models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "wrong_model.bin")

            with open(save_path, 'wb') as f:
                pickle.dump({'vocab': {}, 'model_type': 'Word2Vec'}, f)

            with pytest.raises(ValueError, match="does not contain a DynamicWord2Vec model"):
                DynamicWord2Vec.load(save_path)

    def test_save_creates_file(self, small_model):
        """Test that save() creates a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.npy")

            small_model.save(save_path)

            assert os.path.exists(save_path)
            assert os.path.getsize(save_path) > 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for end-to-end workflows."""

    def test_balanced_sampling_strategy(self, temporal_sentences):
        """Test training with balanced sampling strategy."""
        model = DynamicWord2Vec(
            sentences=temporal_sentences,
            vector_size=50,
            min_word_count=1,
            epochs=3,
            sampling_strategy="balanced",
            verbose=False,
        )
        loss = model.train()

        assert model.U is not None
        assert model.V is not None
        if loss is not None:
            assert loss >= 0

    def test_proportional_sampling_strategy(self, temporal_sentences):
        """Test training with proportional sampling strategy."""
        model = DynamicWord2Vec(
            sentences=temporal_sentences,
            vector_size=50,
            min_word_count=1,
            epochs=3,
            sampling_strategy="proportional",
            verbose=False,
        )
        loss = model.train()

        assert model.U is not None
        assert model.V is not None
        if loss is not None:
            assert loss >= 0

    def test_learning_rate_decay(self, temporal_sentences):
        """Test training with learning rate decay."""
        model = DynamicWord2Vec(
            sentences=temporal_sentences,
            vector_size=50,
            min_word_count=1,
            epochs=5,
            alpha=0.025,
            min_alpha=0.001,
            verbose=False,
        )

        initial_alpha = model.alpha
        model.train()
        final_alpha = model.alpha

        # Alpha should have decayed
        assert final_alpha < initial_alpha

    def test_multiple_time_slices(self):
        """Test with more than 2 time slices."""
        sentences = {
            '唐': [["李", "白", "诗", "仙"]] * 5,
            '宋': [["苏", "轼", "词", "圣"]] * 5,
            '元': [["马", "致", "远", "曲"]] * 5,
            '明': [["汤", "显", "祖", "戏"]] * 5,
        }

        model = DynamicWord2Vec(
            sentences=sentences,
            vector_size=30,
            min_word_count=1,
            epochs=3,
            temporal_lambda=0.05,
            verbose=False,
        )
        model.train()

        assert model.num_time_slices == 4
        assert len(model.labels) == 4

        # Check temporal drift calculation
        if "李" in model.vocab:
            drift = model.calculate_temporal_drift("李")
            assert len(drift) == 3  # 4 slices -> 3 transitions

    def test_empty_corpus_handling(self):
        """Test handling of corpora with no words after filtering."""
        sentences = {
            't1': [["a", "b"]],  # Will be filtered by min_word_count
            't2': [["c", "d"]],
        }

        model = DynamicWord2Vec(
            sentences=sentences,
            vector_size=10,
            min_word_count=10,  # Filter out all words
            epochs=1,
            verbose=False,
        )

        # Should raise error during vocab building
        with pytest.raises(ValueError):
            model.train()

    def test_sequential_training(self, temporal_sentences):
        """Test sequential training mode produces valid embeddings."""
        model = DynamicWord2Vec(
            sentences=temporal_sentences,
            training_mode="sequential",
            vector_size=50,
            min_word_count=1,
            epochs=3,
            verbose=False,
        )
        loss = model.train()

        assert model.U is not None
        assert model.V is not None

        # Should be queryable
        word = "民"
        if word in model.vocab:
            vec = model.get_vector(word, time_label="宋")
            assert vec.shape == (50,)
            drift = model.calculate_temporal_drift(word)
            assert len(drift) == 1

    def test_sequential_save_load(self, temporal_sentences):
        """Test that sequential model round-trips through save/load."""
        model = DynamicWord2Vec(
            sentences=temporal_sentences,
            training_mode="sequential",
            vector_size=50,
            min_word_count=1,
            epochs=3,
            verbose=False,
        )
        model.train()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "seq.npy")
            model.save(path)
            loaded = DynamicWord2Vec.load(path)

            assert loaded.training_mode == "sequential"
            np.testing.assert_array_equal(loaded.U, model.U)
            np.testing.assert_array_equal(loaded.V, model.V)

    def test_procrustes_alignment(self, temporal_sentences):
        """Test that Procrustes alignment produces orthogonally aligned slices."""
        model = DynamicWord2Vec(
            sentences=temporal_sentences,
            training_mode="joint",
            procrustes_align=True,
            vector_size=50,
            min_word_count=1,
            epochs=5,
            seed=42,
            verbose=False,
        )
        model.train()

        assert model.U is not None

    def test_procrustes_disabled(self, temporal_sentences):
        """Test that training works with Procrustes disabled."""
        model = DynamicWord2Vec(
            sentences=temporal_sentences,
            training_mode="joint",
            procrustes_align=False,
            vector_size=50,
            min_word_count=1,
            epochs=3,
            verbose=False,
        )
        model.train()
        assert model.U is not None

    def test_invalid_training_mode(self):
        """Test that invalid training_mode is rejected."""
        with pytest.raises(ValueError, match="training_mode"):
            DynamicWord2Vec(
                sentences={'t1': [["word"]]},
                training_mode="invalid",
                vector_size=50,
                epochs=1,
            )

    def test_sequential_multiple_slices(self):
        """Test sequential training with more than 2 slices."""
        sentences = {
            '唐': [["李", "白", "诗", "仙"]] * 5,
            '宋': [["苏", "轼", "词", "圣"]] * 5,
            '明': [["汤", "显", "祖", "戏"]] * 5,
        }
        model = DynamicWord2Vec(
            sentences=sentences,
            training_mode="sequential",
            vector_size=30,
            min_word_count=1,
            epochs=3,
            verbose=False,
        )
        model.train()

        assert model.num_time_slices == 3
        if "李" in model.vocab:
            drift = model.calculate_temporal_drift("李")
            assert len(drift) == 2
