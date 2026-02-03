"""
Tests for qhchina.analytics.perplexity module.
"""
import pytest
import numpy as np


class TestVisualizePerplexities:
    """Tests for the visualize_perplexities function."""
    
    def test_visualize_perplexities_basic(self):
        """Test basic perplexity visualization."""
        import matplotlib
        matplotlib.use('Agg')  # Use non-GUI backend
        import matplotlib.pyplot as plt
        from qhchina.analytics.perplexity import visualize_perplexities
        from qhchina import helpers
        
        helpers.load_fonts()
        perplexities = [10.0, 20.0, 15.0, 30.0]
        labels = ['a', 'b', 'c', 'd']
        
        # Should not raise
        visualize_perplexities(perplexities, labels)
        plt.close('all')
    
    def test_visualize_perplexities_mismatched_lengths(self):
        """Test that mismatched lengths raise error."""
        from qhchina.analytics.perplexity import visualize_perplexities
        
        perplexities = [10.0, 20.0, 15.0]
        labels = ['a', 'b']  # Mismatched length
        
        with pytest.raises(ValueError, match="Number of labels"):
            visualize_perplexities(perplexities, labels)
    
    def test_visualize_perplexities_with_custom_params(self):
        """Test visualization with custom parameters."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from qhchina.analytics.perplexity import visualize_perplexities
        from qhchina import helpers
        
        helpers.load_fonts()
        perplexities = [5.0, 10.0, 8.0]
        labels = ['x', 'y', 'z']
        
        # Should not raise
        visualize_perplexities(
            perplexities, 
            labels, 
            width=10, 
            height=4, 
            color='blue'
        )
        plt.close('all')


class TestCalculatePerplexityOfTokens:
    """Tests for calculate_perplexity_of_tokens function."""
    
    @pytest.fixture
    def torch_available(self):
        """Check if PyTorch is available."""
        try:
            import torch
            return True
        except ImportError:
            return False
    
    def test_calculate_perplexity_import_error(self, torch_available):
        """Test that ImportError is raised when torch is not installed."""
        if torch_available:
            pytest.skip("PyTorch is installed, cannot test ImportError")
        
        from qhchina.analytics.perplexity import calculate_perplexity_of_tokens
        
        with pytest.raises(ImportError, match="PyTorch is required"):
            calculate_perplexity_of_tokens(None, None, "test")


class TestCalculateWordPerplexity:
    """Tests for calculate_word_perplexity function."""
    
    @pytest.fixture
    def torch_available(self):
        """Check if PyTorch is available."""
        try:
            import torch
            return True
        except ImportError:
            return False
    
    def test_calculate_word_perplexity_import_error(self, torch_available):
        """Test that ImportError is raised when torch is not installed."""
        if torch_available:
            pytest.skip("PyTorch is installed, cannot test ImportError")
        
        from qhchina.analytics.perplexity import calculate_word_perplexity
        
        with pytest.raises(ImportError, match="PyTorch is required"):
            calculate_word_perplexity(None, None, "context", "word")
