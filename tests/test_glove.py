"""
Tests for qhchina.analytics.embeddings.GloVe.
"""

import os
import tempfile

import numpy as np
import pytest


class TestGloVeBasic:
    def test_train_in_memory(self, sample_documents):
        from qhchina.analytics.embeddings import GloVe

        model = GloVe(
            sentences=sample_documents,
            vector_size=20,
            window=2,
            min_word_count=1,
            epochs=2,
            seed=42,
            mode="in_memory",
            verbose=False,
        )
        loss = model.train(epochs=model.epochs)
        assert isinstance(loss, float)
        assert len(model.vocab) > 0
        assert model.W.shape[1] == 20

    def test_train_disk_mode(self, sample_documents):
        from qhchina.analytics.embeddings import GloVe

        model = GloVe(
            sentences=sample_documents,
            vector_size=16,
            window=2,
            min_word_count=1,
            epochs=2,
            seed=42,
            mode="disk",
            shard_sentence_count=2,
            verbose=False,
        )
        loss = model.train(epochs=model.epochs)
        assert isinstance(loss, float)
        assert model.W.shape[1] == 16

    def test_invalid_mode_raises(self):
        from qhchina.analytics.embeddings import GloVe

        with pytest.raises(ValueError, match="mode"):
            GloVe(vector_size=20, epochs=1, mode="invalid")

    def test_rejects_one_shot_iterators(self):
        from qhchina.analytics.embeddings import GloVe

        sentences = iter([["甲", "乙"], ["乙", "丙"]])
        model = GloVe(vector_size=10, min_word_count=1, epochs=1)
        with pytest.raises(ValueError, match="restartable"):
            model.train(sentences=sentences)

    def test_train_requires_epochs_somewhere(self, sample_documents):
        from qhchina.analytics.embeddings import GloVe

        model = GloVe(
            sentences=sample_documents,
            vector_size=10,
            window=2,
            min_word_count=1,
            epochs=1,
            mode="in_memory",
        )
        # Allowed because epochs is set on the model.
        model.train()

        model_no_epochs = GloVe(
            vector_size=10,
            window=2,
            min_word_count=1,
            epochs=1,
            mode="in_memory",
        )
        model_no_epochs.epochs = None
        with pytest.raises(ValueError, match="epochs must be specified either at init or in train"):
            model_no_epochs.train(sentences=sample_documents)

    def test_disk_mode_streams_with_memmap_shards(self, sample_documents, monkeypatch):
        from qhchina.analytics.embeddings import GloVe
        from qhchina.analytics.embeddings.glove import base as glove_base

        real_np_load = glove_base.np.load
        mmap_calls: list[object] = []

        def load_proxy(*args, **kwargs):
            mmap_calls.append(kwargs.get("mmap_mode"))
            return real_np_load(*args, **kwargs)

        monkeypatch.setattr(glove_base.np, "load", load_proxy)

        model = GloVe(
            sentences=sample_documents,
            vector_size=12,
            window=2,
            min_word_count=1,
            epochs=2,
            seed=42,
            mode="disk",
            shard_sentence_count=2,
            cooc_train_chunk_size=8,
            max_cooc_entries_in_memory=4,
            verbose=False,
        )
        loss = model.train(epochs=model.epochs)
        assert isinstance(loss, float)
        assert "r" in mmap_calls


class TestGloVeVectors:
    @pytest.fixture
    def trained_glove(self, larger_documents):
        from qhchina.analytics.embeddings import GloVe

        model = GloVe(
            sentences=larger_documents,
            vector_size=24,
            window=3,
            min_word_count=2,
            epochs=2,
            seed=7,
            mode="in_memory",
            verbose=False,
        )
        model.train(epochs=model.epochs)
        return model

    def test_get_vector(self, trained_glove):
        word = next(iter(trained_glove.vocab.keys()))
        vec = trained_glove.get_vector(word)
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (trained_glove.vector_size,)

    def test_similarity(self, trained_glove):
        words = list(trained_glove.vocab.keys())
        if len(words) >= 2:
            sim = trained_glove.similarity(words[0], words[1])
            assert isinstance(sim, float)
            assert -1.0 <= sim <= 1.0

    def test_most_similar(self, trained_glove):
        word = next(iter(trained_glove.vocab.keys()))
        result = trained_glove.most_similar(word, topn=5)
        assert isinstance(result, list)
        assert len(result) <= 5


class TestGloVePersistence:
    def test_save_load_roundtrip(self, sample_documents):
        from qhchina.analytics.embeddings import GloVe

        model = GloVe(
            sentences=sample_documents,
            vector_size=12,
            window=2,
            min_word_count=1,
            epochs=2,
            seed=21,
            mode="in_memory",
            verbose=False,
        )
        model.train(epochs=model.epochs)

        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as handle:
            path = handle.name
        try:
            model.save(path)
            loaded = GloVe.load(path)
            assert loaded.vector_size == model.vector_size
            assert loaded.vocab == model.vocab
            np.testing.assert_allclose(loaded.W, model.W, rtol=1e-6, atol=1e-6)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_export_glove_format(self, sample_documents):
        from qhchina.analytics.embeddings import GloVe

        model = GloVe(
            sentences=sample_documents,
            vector_size=10,
            window=2,
            min_word_count=1,
            epochs=1,
            seed=11,
            mode="in_memory",
            verbose=False,
        )
        model.train(epochs=model.epochs)

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as handle:
            path = handle.name
        try:
            model.export(path, format="glove")
            with open(path, "r", encoding="utf-8") as reader:
                lines = [line.strip() for line in reader if line.strip()]
            assert len(lines) == len(model.vocab)
        finally:
            if os.path.exists(path):
                os.unlink(path)


class TestGloVeReproducibility:
    def test_same_seed_reproducible(self, sample_documents):
        from qhchina.analytics.embeddings import GloVe

        kwargs = dict(
            sentences=sample_documents,
            vector_size=14,
            window=2,
            min_word_count=1,
            epochs=2,
            seed=123,
            mode="in_memory",
            verbose=False,
        )
        model_a = GloVe(**kwargs)
        model_b = GloVe(**kwargs)
        model_a.train(epochs=model_a.epochs)
        model_b.train(epochs=model_b.epochs)
        np.testing.assert_allclose(model_a.W, model_b.W, rtol=1e-6, atol=1e-6)

    def test_mode_parity_shapes(self, sample_documents):
        from qhchina.analytics.embeddings import GloVe

        common = dict(
            sentences=sample_documents,
            vector_size=12,
            window=2,
            min_word_count=1,
            epochs=1,
            seed=9,
            verbose=False,
        )
        mem_model = GloVe(**common, mode="in_memory")
        disk_model = GloVe(**common, mode="disk", shard_sentence_count=2)
        mem_model.train(epochs=mem_model.epochs)
        disk_model.train(epochs=disk_model.epochs)

        assert mem_model.W.shape == disk_model.W.shape
        assert mem_model.W.dtype == np.float32
        assert disk_model.W.dtype == np.float32
