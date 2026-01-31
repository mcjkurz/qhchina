"""
Tests for qhchina.utils module.
"""
import pytest


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
