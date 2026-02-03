"""
Tests for qhchina.analytics.vectors module.
"""
import pytest
import numpy as np


# =============================================================================
# Cosine Similarity Tests
# =============================================================================

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
    
    def test_cosine_similarity_zero_vector(self):
        """Test cosine similarity with zero vector returns 0."""
        from qhchina.analytics.vectors import cosine_similarity
        
        vec_a = np.array([1.0, 2.0, 3.0])
        vec_zero = np.array([0.0, 0.0, 0.0])
        
        sim = cosine_similarity(vec_a, vec_zero)
        
        assert sim == 0.0
    
    def test_cosine_similarity_list_input(self):
        """Test cosine similarity accepts list input."""
        from qhchina.analytics.vectors import cosine_similarity
        
        vec_a = [1.0, 2.0, 3.0]
        vec_b = [1.0, 2.0, 3.0]
        
        sim = cosine_similarity(vec_a, vec_b)
        
        assert sim == pytest.approx(1.0)
    
    def test_cosine_similarity_matrix_input(self):
        """Test cosine similarity with matrix input."""
        from qhchina.analytics.vectors import cosine_similarity
        
        vectors1 = np.array([[1.0, 0.0], [0.0, 1.0]])
        vectors2 = np.array([[1.0, 0.0], [0.0, 1.0]])
        
        sim_matrix = cosine_similarity(vectors1, vectors2)
        
        assert sim_matrix.shape == (2, 2)
        assert sim_matrix[0, 0] == pytest.approx(1.0)
        assert sim_matrix[0, 1] == pytest.approx(0.0)


class TestCosineDistance:
    """Tests for cosine distance function."""
    
    def test_cosine_distance_identical(self):
        """Test cosine distance of identical vectors is 0."""
        from qhchina.analytics.vectors import cosine_distance
        
        vec = np.array([1.0, 2.0, 3.0])
        
        dist = cosine_distance(vec, vec)
        
        assert dist == pytest.approx(0.0)
    
    def test_cosine_distance_opposite(self):
        """Test cosine distance of opposite vectors is 2."""
        from qhchina.analytics.vectors import cosine_distance
        
        vec_a = np.array([1.0, 0.0])
        vec_b = np.array([-1.0, 0.0])
        
        dist = cosine_distance(vec_a, vec_b)
        
        assert dist == pytest.approx(2.0)
    
    def test_cosine_distance_orthogonal(self):
        """Test cosine distance of orthogonal vectors is 1."""
        from qhchina.analytics.vectors import cosine_distance
        
        vec_a = np.array([1.0, 0.0])
        vec_b = np.array([0.0, 1.0])
        
        dist = cosine_distance(vec_a, vec_b)
        
        assert dist == pytest.approx(1.0)


class TestProject2D:
    """Tests for 2D projection function."""
    
    def test_project_2d_pca(self):
        """Test PCA projection."""
        import matplotlib
        matplotlib.use('Agg')
        from qhchina.analytics.vectors import project_2d
        from qhchina import helpers
        import matplotlib.pyplot as plt
        
        helpers.load_fonts()
        
        # Create some random vectors
        vectors = np.random.randn(10, 50)
        labels = [f"vec_{i}" for i in range(10)]
        
        # Should not raise
        project_2d(vectors, labels=labels, method='pca')
        plt.close('all')
    
    def test_project_2d_tsne(self):
        """Test t-SNE projection."""
        import matplotlib
        matplotlib.use('Agg')
        from qhchina.analytics.vectors import project_2d
        from qhchina import helpers
        import matplotlib.pyplot as plt
        
        helpers.load_fonts()
        vectors = np.random.randn(10, 50)
        labels = [f"vec_{i}" for i in range(10)]
        
        # t-SNE requires perplexity
        project_2d(vectors, labels=labels, method='tsne', perplexity=3)
        plt.close('all')
    
    def test_project_2d_dict_input(self):
        """Test projection with dict input."""
        import matplotlib
        matplotlib.use('Agg')
        from qhchina.analytics.vectors import project_2d
        from qhchina import helpers
        import matplotlib.pyplot as plt
        
        helpers.load_fonts()
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
    
    def test_project_2d_umap(self):
        """Test UMAP projection if available."""
        import matplotlib
        matplotlib.use('Agg')
        from qhchina.analytics.vectors import project_2d
        from qhchina import helpers
        import matplotlib.pyplot as plt
        
        helpers.load_fonts()
        try:
            import umap
        except ImportError:
            pytest.skip("umap-learn not installed")
        
        vectors = np.random.randn(10, 50)
        labels = [f"vec_{i}" for i in range(10)]
        
        project_2d(vectors, labels=labels, method='umap')
        plt.close('all')
    
    def test_project_2d_mismatched_labels(self):
        """Test that mismatched labels raise error."""
        from qhchina.analytics.vectors import project_2d
        
        vectors = np.random.randn(10, 50)
        labels = [f"vec_{i}" for i in range(5)]  # Wrong number of labels
        
        with pytest.raises(ValueError, match="Number of labels must match"):
            project_2d(vectors, labels=labels, method='pca')
    
    def test_project_2d_save_to_file(self, tmp_path):
        """Test saving projection to file."""
        import matplotlib
        matplotlib.use('Agg')
        from qhchina.analytics.vectors import project_2d
        from qhchina import helpers
        import matplotlib.pyplot as plt
        
        helpers.load_fonts()
        vectors = np.random.randn(10, 50)
        labels = [f"vec_{i}" for i in range(10)]
        
        filepath = tmp_path / "projection.png"
        project_2d(vectors, labels=labels, method='pca', filename=str(filepath))
        plt.close('all')
        
        assert filepath.exists()
    
    def test_project_2d_with_colors(self):
        """Test projection with custom colors."""
        import matplotlib
        matplotlib.use('Agg')
        from qhchina.analytics.vectors import project_2d
        from qhchina import helpers
        import matplotlib.pyplot as plt
        
        helpers.load_fonts()
        vectors = np.random.randn(5, 50)
        labels = [f"vec_{i}" for i in range(5)]
        colors = ['red', 'blue', 'green', 'yellow', 'purple']
        
        project_2d(vectors, labels=labels, method='pca', color=colors)
        plt.close('all')
    
    def test_project_2d_with_title(self):
        """Test projection with title."""
        import matplotlib
        matplotlib.use('Agg')
        from qhchina.analytics.vectors import project_2d
        from qhchina import helpers
        import matplotlib.pyplot as plt
        
        helpers.load_fonts()
        vectors = np.random.randn(5, 50)
        
        project_2d(vectors, method='pca', title="Test Projection")
        plt.close('all')


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


class TestAlignVectors:
    """Tests for vector alignment using Procrustes analysis."""
    
    def test_align_vectors_basic(self):
        """Test basic vector alignment."""
        from qhchina.analytics.vectors import align_vectors
        
        # Create source and target vectors
        source = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        target = np.array([[0.0, 1.0], [-1.0, 0.0], [-1.0, 1.0]])
        
        aligned, rotation = align_vectors(source, target)
        
        assert aligned.shape == source.shape
        assert rotation.shape == (2, 2)
    
    def test_align_vectors_returns_orthogonal_matrix(self):
        """Test that the rotation matrix is orthogonal."""
        from qhchina.analytics.vectors import align_vectors
        
        np.random.seed(42)
        source = np.random.randn(10, 5)
        target = np.random.randn(10, 5)
        
        _, rotation = align_vectors(source, target)
        
        # Orthogonal matrix: R @ R.T = I
        identity = np.dot(rotation, rotation.T)
        assert np.allclose(identity, np.eye(rotation.shape[0]), atol=1e-10)
    
    def test_align_vectors_improves_similarity(self):
        """Test that alignment improves similarity between source and target."""
        from qhchina.analytics.vectors import align_vectors
        
        np.random.seed(42)
        # Create a rotation matrix
        theta = np.pi / 4  # 45 degree rotation
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        
        # Source vectors
        source = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5]])
        # Target is rotated source (plus some noise)
        target = np.dot(source, R.T) + np.random.randn(*source.shape) * 0.01
        
        aligned, _ = align_vectors(source, target)
        
        # The aligned vectors should be closer to target than original source
        original_dist = np.mean(np.sum((source - target) ** 2, axis=1))
        aligned_dist = np.mean(np.sum((aligned - target) ** 2, axis=1))
        
        assert aligned_dist < original_dist


# =============================================================================
# Most Similar Edge Cases
# =============================================================================

class TestMostSimilarEdgeCases:
    """Tests for edge cases in most_similar function."""
    
    def test_most_similar_top_n_larger_than_vectors(self):
        """Test most_similar when top_n is larger than number of vectors."""
        from qhchina.analytics.vectors import most_similar
        
        target = np.array([1.0, 0.0])
        vectors = np.array([[0.9, 0.1], [0.0, 1.0]])
        
        # Request more results than available
        result = most_similar(target, vectors, top_n=10)
        
        # Should return all available (2)
        assert len(result) == 2
    
    def test_most_similar_custom_metric(self):
        """Test most_similar with custom metric function."""
        from qhchina.analytics.vectors import most_similar
        
        target = np.array([1.0, 0.0])
        vectors = np.array([[0.9, 0.1], [0.0, 1.0]])
        
        # Custom metric: negative Euclidean distance (to get "similarity")
        def neg_euclidean(v1, v2):
            return -np.linalg.norm(v1 - v2)
        
        result = most_similar(target, vectors, metric=neg_euclidean, top_n=1)
        
        assert len(result) == 1
    
    def test_most_similar_invalid_metric(self):
        """Test that invalid metric raises error."""
        from qhchina.analytics.vectors import most_similar
        
        target = np.array([1.0, 0.0])
        vectors = np.array([[0.9, 0.1]])
        
        with pytest.raises(ValueError, match="metric must be"):
            most_similar(target, vectors, metric='invalid')
    
    def test_most_similar_mismatched_labels(self):
        """Test that mismatched labels raise error."""
        from qhchina.analytics.vectors import most_similar
        
        target = np.array([1.0, 0.0])
        vectors = np.array([[0.9, 0.1], [0.0, 1.0]])
        labels = ["only_one_label"]  # Wrong number
        
        with pytest.raises(ValueError, match="Number of labels must match"):
            most_similar(target, vectors, labels=labels)


# =============================================================================
# Bias Direction Edge Cases
# =============================================================================

class TestBiasDirectionEdgeCases:
    """Tests for edge cases in bias direction functions."""
    
    def test_get_bias_direction_identical_anchors(self):
        """Test bias direction when anchors are identical (zero direction)."""
        from qhchina.analytics.vectors import get_bias_direction
        
        # Same vectors -> zero difference
        pos_anchor = np.array([1.0, 0.0])
        neg_anchor = np.array([1.0, 0.0])
        
        direction = get_bias_direction((pos_anchor, neg_anchor))
        
        # Should handle gracefully (normalized zero or small value)
        assert isinstance(direction, np.ndarray)
        assert len(direction) == 2
    
    def test_calculate_bias_missing_word(self):
        """Test calculate_bias with missing target word."""
        from qhchina.analytics.vectors import calculate_bias
        
        class MockWordVectors:
            def __init__(self):
                self.vectors = {
                    "he": np.array([1.0, 0.0]),
                    "she": np.array([0.0, 1.0]),
                }
            
            def __getitem__(self, word):
                if word not in self.vectors:
                    raise KeyError(f"Word '{word}' not in vocabulary")
                return self.vectors[word]
            
            def __contains__(self, word):
                return word in self.vectors
        
        word_vectors = MockWordVectors()
        
        with pytest.raises(KeyError):
            calculate_bias(
                anchors=("he", "she"),
                targets=["missing_word"],
                word_vectors=word_vectors
            )


# =============================================================================
# Align Vectors Edge Cases
# =============================================================================

class TestAlignVectorsEdgeCases:
    """Tests for edge cases in align_vectors function."""
    
    def test_align_vectors_single_vector(self):
        """Test alignment with single vectors."""
        from qhchina.analytics.vectors import align_vectors
        
        source = np.array([[1.0, 0.0]])
        target = np.array([[0.0, 1.0]])
        
        aligned, rotation = align_vectors(source, target)
        
        assert aligned.shape == source.shape
    
    def test_align_vectors_high_dimensional(self):
        """Test alignment with high dimensional vectors."""
        from qhchina.analytics.vectors import align_vectors
        
        np.random.seed(42)
        source = np.random.randn(20, 100)  # 20 vectors, 100 dimensions
        target = np.random.randn(20, 100)
        
        aligned, rotation = align_vectors(source, target)
        
        assert aligned.shape == source.shape
        assert rotation.shape == (100, 100)
    
    def test_align_vectors_preserves_distances(self):
        """Test that Procrustes preserves pairwise distances in source."""
        from qhchina.analytics.vectors import align_vectors
        from scipy.spatial.distance import pdist
        
        np.random.seed(42)
        source = np.random.randn(5, 10)
        target = np.random.randn(5, 10)
        
        aligned, _ = align_vectors(source, target)
        
        # Pairwise distances should be preserved (rotation is distance-preserving)
        source_dists = pdist(source)
        aligned_dists = pdist(aligned)
        
        assert np.allclose(source_dists, aligned_dists, rtol=1e-10)
