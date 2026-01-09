"""
Tests for qhchina.analytics.vectors module.
"""
import pytest
import numpy as np


class TestCosineSimilarity:
    """Tests for cosine similarity function."""
    
    def test_cosine_similarity_identical(self):
        """Test cosine similarity of identical vectors."""
        from qhchina.analytics.vectors import cosine_similarity
        
        vec = np.array([1.0, 2.0, 3.0])
        
        sim = cosine_similarity(vec, vec)
        
        assert sim == pytest.approx(1.0)
    
    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity of orthogonal vectors."""
        from qhchina.analytics.vectors import cosine_similarity
        
        vec_a = np.array([1.0, 0.0, 0.0])
        vec_b = np.array([0.0, 1.0, 0.0])
        
        sim = cosine_similarity(vec_a, vec_b)
        
        assert sim == pytest.approx(0.0)
    
    def test_cosine_similarity_opposite(self):
        """Test cosine similarity of opposite vectors."""
        from qhchina.analytics.vectors import cosine_similarity
        
        vec_a = np.array([1.0, 2.0, 3.0])
        vec_b = np.array([-1.0, -2.0, -3.0])
        
        sim = cosine_similarity(vec_a, vec_b)
        
        assert sim == pytest.approx(-1.0)


class TestProject2D:
    """Tests for 2D projection function."""
    
    def test_project_2d_pca(self):
        """Test PCA projection."""
        from qhchina.analytics.vectors import project_2d
        import matplotlib.pyplot as plt
        
        # Create some random vectors
        vectors = np.random.randn(10, 50)
        labels = [f"vec_{i}" for i in range(10)]
        
        # Should not raise
        project_2d(vectors, labels=labels, method='pca')
        plt.close('all')
    
    def test_project_2d_tsne(self):
        """Test t-SNE projection."""
        from qhchina.analytics.vectors import project_2d
        import matplotlib.pyplot as plt
        
        vectors = np.random.randn(10, 50)
        labels = [f"vec_{i}" for i in range(10)]
        
        # t-SNE requires perplexity
        project_2d(vectors, labels=labels, method='tsne', perplexity=3)
        plt.close('all')
    
    def test_project_2d_dict_input(self):
        """Test projection with dict input."""
        from qhchina.analytics.vectors import project_2d
        import matplotlib.pyplot as plt
        
        vectors = {
            "a": np.random.randn(50),
            "b": np.random.randn(50),
            "c": np.random.randn(50),
        }
        
        project_2d(vectors, method='pca')
        plt.close('all')
    
    def test_project_2d_tsne_no_perplexity(self):
        """Test that t-SNE without perplexity raises error."""
        from qhchina.analytics.vectors import project_2d
        
        vectors = np.random.randn(10, 50)
        
        with pytest.raises(ValueError):
            project_2d(vectors, method='tsne')
    
    def test_project_2d_invalid_method(self):
        """Test that invalid method raises error."""
        from qhchina.analytics.vectors import project_2d
        
        vectors = np.random.randn(10, 50)
        
        with pytest.raises(ValueError):
            project_2d(vectors, method='invalid')


class TestMostSimilar:
    """Tests for most_similar function."""
    
    def test_most_similar_basic(self):
        """Test finding most similar vectors."""
        from qhchina.analytics.vectors import most_similar
        
        # Target vector
        target = np.array([1.0, 1.0, 0.0])
        
        # Vectors to compare against
        vectors = np.array([
            [0.9, 1.1, 0.1],  # Most similar to target
            [1.0, 0.0, 0.0],  # Less similar
            [0.0, 0.0, 1.0],  # Orthogonal
        ])
        labels = ["similar", "less_similar", "orthogonal"]
        
        result = most_similar(target, vectors, labels=labels, top_n=2)
        
        assert isinstance(result, list)
        assert len(result) == 2
        # Most similar should be first
        assert result[0][0] == "similar"
    
    def test_most_similar_without_labels(self):
        """Test most_similar without labels (returns indices)."""
        from qhchina.analytics.vectors import most_similar
        
        target = np.array([1.0, 0.0])
        vectors = np.array([[0.9, 0.1], [0.0, 1.0]])
        
        result = most_similar(target, vectors, top_n=1)
        
        assert isinstance(result, list)
        # Returns (index, similarity)
        assert result[0][0] == 0  # First vector is most similar


class TestGetBiasDirection:
    """Tests for bias direction calculation."""
    
    def test_get_bias_direction_single_pair(self):
        """Test computing bias direction from a single pair."""
        from qhchina.analytics.vectors import get_bias_direction
        
        pos_anchor = np.array([1.0, 0.0])
        neg_anchor = np.array([0.0, 1.0])
        
        direction = get_bias_direction((pos_anchor, neg_anchor))
        
        assert isinstance(direction, np.ndarray)
        assert len(direction) == 2
    
    def test_get_bias_direction_multiple_pairs(self):
        """Test computing bias direction from multiple pairs."""
        from qhchina.analytics.vectors import get_bias_direction
        
        pairs = [
            (np.array([1.0, 0.0]), np.array([0.0, 1.0])),
            (np.array([0.9, 0.1]), np.array([0.1, 0.9])),
        ]
        
        direction = get_bias_direction(pairs)
        
        assert isinstance(direction, np.ndarray)
        assert len(direction) == 2


class TestCalculateBias:
    """Tests for bias calculation using word vectors."""
    
    def test_calculate_bias_with_word_vectors(self):
        """Test computing bias for words along a direction using word vectors."""
        from qhchina.analytics.vectors import calculate_bias
        
        # Mock word_vectors object (dict-like)
        class MockWordVectors:
            def __init__(self):
                self.vectors = {
                    "he": np.array([1.0, 0.0]),
                    "she": np.array([0.0, 1.0]),
                    "doctor": np.array([0.8, 0.2]),
                    "nurse": np.array([0.2, 0.8]),
                }
            
            def __getitem__(self, word):
                return self.vectors[word]
            
            def __contains__(self, word):
                return word in self.vectors
        
        word_vectors = MockWordVectors()
        
        # Calculate bias using anchor pairs and target words
        biases = calculate_bias(
            anchors=("he", "she"),
            targets=["doctor", "nurse"],
            word_vectors=word_vectors
        )
        
        assert isinstance(biases, np.ndarray)
        assert len(biases) == 2
