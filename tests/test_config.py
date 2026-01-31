"""
Tests for qhchina.config module.
"""
import pytest
import numpy as np


class TestRandomSeed:
    """Tests for random seed management."""
    
    def test_set_get_random_seed(self):
        """Test setting and getting random seed."""
        from qhchina.config import set_random_seed, get_random_seed
        
        # Set a seed
        set_random_seed(42)
        assert get_random_seed() == 42
        
        # Change the seed
        set_random_seed(123)
        assert get_random_seed() == 123
        
        # Reset to None
        set_random_seed(None)
        assert get_random_seed() is None
    
    def test_get_rng_with_seed(self):
        """Test getting RNG with a specific seed."""
        from qhchina.config import get_rng
        
        rng1 = get_rng(42)
        val1 = rng1.random()
        
        rng2 = get_rng(42)
        val2 = rng2.random()
        
        # Same seed should produce same value
        assert val1 == val2
    
    def test_get_rng_uses_global_seed(self):
        """Test that get_rng uses global seed when no local seed provided."""
        from qhchina.config import set_random_seed, get_rng
        
        set_random_seed(42)
        
        rng1 = get_rng()
        val1 = rng1.random()
        
        rng2 = get_rng()
        val2 = rng2.random()
        
        # Same global seed should produce same first value
        assert val1 == val2
        
        # Clean up
        set_random_seed(None)
    
    def test_get_rng_local_overrides_global(self):
        """Test that local seed overrides global seed."""
        from qhchina.config import set_random_seed, get_rng
        
        set_random_seed(42)
        
        rng_local = get_rng(123)
        val_local = rng_local.random()
        
        rng_global = get_rng()
        val_global = rng_global.random()
        
        # Different seeds should produce different values
        assert val_local != val_global
        
        # Clean up
        set_random_seed(None)
    
    def test_get_rng_returns_random_state(self):
        """Test that get_rng returns a numpy RandomState."""
        from qhchina.config import get_rng
        
        rng = get_rng(42)
        assert isinstance(rng, np.random.RandomState)
    
    def test_resolve_seed_local_priority(self):
        """Test that local seed has highest priority."""
        from qhchina.config import resolve_seed, set_random_seed
        
        set_random_seed(42)  # Global seed
        
        # Local seed should override global
        assert resolve_seed(123) == 123
        
        # Clean up
        set_random_seed(None)
    
    def test_resolve_seed_default_priority(self):
        """Test that default seed is used when no local seed."""
        from qhchina.config import resolve_seed, set_random_seed
        
        set_random_seed(42)  # Global seed
        
        # Default should override global when local is None
        assert resolve_seed(None, default_seed=100) == 100
        
        # Clean up
        set_random_seed(None)
    
    def test_resolve_seed_global_fallback(self):
        """Test that global seed is used as fallback."""
        from qhchina.config import resolve_seed, set_random_seed
        
        set_random_seed(42)
        
        # With no local or default, should use global
        assert resolve_seed(None) == 42
        
        # Clean up
        set_random_seed(None)
    
    def test_resolve_seed_all_none(self):
        """Test that None is returned when all seeds are None."""
        from qhchina.config import resolve_seed, set_random_seed
        
        set_random_seed(None)
        
        assert resolve_seed(None) is None
        assert resolve_seed(None, default_seed=None) is None
