"""
Tests for qhchina.utils module.
"""
import pytest
import numpy as np


class TestValidateFilters:
    """Tests for the validate_filters function."""
    
    def test_validate_filters_none(self):
        """Test that None filters pass validation."""
        from qhchina.utils import validate_filters
        
        # Should not raise
        validate_filters(None, {'key1', 'key2'}, context='test')
    
    def test_validate_filters_valid_keys(self):
        """Test that valid filter keys pass validation."""
        from qhchina.utils import validate_filters
        
        filters = {'key1': 'value1', 'key2': 'value2'}
        valid_keys = {'key1', 'key2', 'key3'}
        
        # Should not raise
        validate_filters(filters, valid_keys, context='test')
    
    def test_validate_filters_invalid_keys(self):
        """Test that invalid filter keys raise ValueError."""
        from qhchina.utils import validate_filters
        
        filters = {'key1': 'value1', 'invalid_key': 'value2'}
        valid_keys = {'key1', 'key2'}
        
        with pytest.raises(ValueError, match="Unknown filter keys"):
            validate_filters(filters, valid_keys, context='test')
    
    def test_validate_filters_non_dict(self):
        """Test that non-dict filters raise TypeError."""
        from qhchina.utils import validate_filters
        
        with pytest.raises(TypeError, match="filters must be a dictionary"):
            validate_filters(['list', 'of', 'things'], {'key1'}, context='test')
    
    def test_validate_filters_empty_dict(self):
        """Test that empty dict passes validation."""
        from qhchina.utils import validate_filters
        
        # Should not raise
        validate_filters({}, {'key1', 'key2'}, context='test')
    
    def test_validate_filters_error_message_includes_context(self):
        """Test that error message includes the context."""
        from qhchina.utils import validate_filters
        
        filters = {'bad_key': 'value'}
        valid_keys = {'good_key'}
        
        with pytest.raises(ValueError) as exc_info:
            validate_filters(filters, valid_keys, context='my_function')
        
        assert 'my_function' in str(exc_info.value)
        assert 'bad_key' in str(exc_info.value)


class TestApplyPValueCorrection:
    """Tests for the apply_p_value_correction function."""
    
    def test_bonferroni_basic(self):
        """Test Bonferroni correction multiplies by n."""
        from qhchina.utils import apply_p_value_correction
        
        p_values = [0.01, 0.04, 0.03]
        adjusted = apply_p_value_correction(p_values, method='bonferroni')
        
        assert len(adjusted) == 3
        np.testing.assert_allclose(adjusted, [0.03, 0.12, 0.09])
    
    def test_bonferroni_capped_at_one(self):
        """Test Bonferroni correction caps p-values at 1.0."""
        from qhchina.utils import apply_p_value_correction
        
        p_values = [0.5, 0.8, 0.01]
        adjusted = apply_p_value_correction(p_values, method='bonferroni')
        
        assert all(a <= 1.0 for a in adjusted)
        np.testing.assert_allclose(adjusted[0], 1.0)
        np.testing.assert_allclose(adjusted[1], 1.0)
    
    def test_fdr_bh_basic(self):
        """Test BH correction produces valid adjusted p-values."""
        from qhchina.utils import apply_p_value_correction
        
        p_values = [0.001, 0.01, 0.04, 0.03, 0.5]
        adjusted = apply_p_value_correction(p_values, method='fdr_bh')
        
        assert len(adjusted) == 5
        # All adjusted p-values should be >= raw p-values
        assert all(a >= p - 1e-15 for a, p in zip(adjusted, p_values))
        # All adjusted p-values should be <= 1.0
        assert all(a <= 1.0 for a in adjusted)
    
    def test_fdr_bh_monotonicity(self):
        """Test BH correction preserves ordering of p-values."""
        from qhchina.utils import apply_p_value_correction
        
        # Sorted p-values
        p_values = [0.001, 0.01, 0.03, 0.04, 0.5]
        adjusted = apply_p_value_correction(p_values, method='fdr_bh')
        
        # Adjusted p-values should also be monotonically non-decreasing
        # when the input is sorted
        for i in range(len(adjusted) - 1):
            assert adjusted[i] <= adjusted[i + 1] + 1e-15
    
    def test_single_p_value(self):
        """Test correction with a single p-value."""
        from qhchina.utils import apply_p_value_correction
        
        p_values = [0.05]
        
        bonf = apply_p_value_correction(p_values, method='bonferroni')
        np.testing.assert_allclose(bonf, [0.05])  # n=1, so no change
        
        fdr = apply_p_value_correction(p_values, method='fdr_bh')
        np.testing.assert_allclose(fdr, [0.05])  # n=1, so no change
    
    def test_empty_p_values(self):
        """Test correction with empty input."""
        from qhchina.utils import apply_p_value_correction
        
        result = apply_p_value_correction([], method='bonferroni')
        assert len(result) == 0
    
    def test_invalid_method_raises_error(self):
        """Test that invalid method raises ValueError."""
        from qhchina.utils import apply_p_value_correction
        
        with pytest.raises(ValueError, match="Unknown correction method"):
            apply_p_value_correction([0.05], method='invalid')
    
    def test_all_ones(self):
        """Test correction with all p-values equal to 1.0."""
        from qhchina.utils import apply_p_value_correction
        
        p_values = [1.0, 1.0, 1.0]
        
        bonf = apply_p_value_correction(p_values, method='bonferroni')
        assert all(a == 1.0 for a in bonf)
        
        fdr = apply_p_value_correction(p_values, method='fdr_bh')
        assert all(a == 1.0 for a in fdr)
    
    def test_all_zeros(self):
        """Test correction with all p-values equal to 0."""
        from qhchina.utils import apply_p_value_correction
        
        p_values = [0.0, 0.0, 0.0]
        
        bonf = apply_p_value_correction(p_values, method='bonferroni')
        assert all(a == 0.0 for a in bonf)
        
        fdr = apply_p_value_correction(p_values, method='fdr_bh')
        assert all(a == 0.0 for a in fdr)
