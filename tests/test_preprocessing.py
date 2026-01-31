"""
Tests for qhchina.preprocessing module.
"""
import pytest
import tempfile
import os


# =============================================================================
# SegmentationWrapper Base Class Tests
# =============================================================================

class TestSegmentationWrapper:
    """Tests for the SegmentationWrapper base class."""
    
    def test_init_default(self):
        """Test default initialization."""
        from qhchina.preprocessing.segmentation import SegmentationWrapper
        
        wrapper = SegmentationWrapper()
        
        assert wrapper.strategy == "whole"
        assert wrapper.chunk_size == 512
    
    def test_init_strategies(self):
        """Test different strategy options."""
        from qhchina.preprocessing.segmentation import SegmentationWrapper
        
        for strategy in ["line", "sentence", "chunk", "whole"]:
            wrapper = SegmentationWrapper(strategy=strategy)
            assert wrapper.strategy == strategy
    
    def test_init_invalid_strategy(self):
        """Test that invalid strategy raises error."""
        from qhchina.preprocessing.segmentation import SegmentationWrapper
        
        with pytest.raises(ValueError):
            SegmentationWrapper(strategy="invalid")
    
    def test_filters_validation(self):
        """Test filter validation."""
        from qhchina.preprocessing.segmentation import SegmentationWrapper
        
        # Valid filters should work
        wrapper = SegmentationWrapper(filters={
            'stopwords': {'的', '了'},
            'min_word_length': 2,
            'excluded_pos': {'x'}
        })
        
        assert wrapper.filters['min_word_length'] == 2
    
    def test_invalid_filter_key(self):
        """Test that invalid filter keys raise error."""
        from qhchina.preprocessing.segmentation import SegmentationWrapper
        
        with pytest.raises(ValueError):
            SegmentationWrapper(filters={'invalid_key': 'value'})
    
    def test_chunk_strategy(self):
        """Test chunk strategy initialization."""
        from qhchina.preprocessing.segmentation import SegmentationWrapper
        
        wrapper = SegmentationWrapper(strategy="chunk", chunk_size=256)
        assert wrapper.strategy == "chunk"
        assert wrapper.chunk_size == 256
    
    def test_custom_chunk_size(self):
        """Test custom chunk_size parameter."""
        from qhchina.preprocessing.segmentation import SegmentationWrapper
        
        wrapper = SegmentationWrapper(strategy="chunk", chunk_size=100)
        assert wrapper.chunk_size == 100


class TestJiebaSegmenter:
    """Tests for Jieba-based segmentation (if available)."""
    
    @pytest.fixture
    def jieba_available(self):
        """Check if jieba is available."""
        try:
            import jieba
            return True
        except ImportError:
            return False
    
    def test_jieba_segment_whole(self, jieba_available, sample_chinese_text):
        """Test whole-text segmentation with Jieba."""
        if not jieba_available:
            pytest.skip("jieba not installed")
        
        from qhchina.preprocessing.segmentation import JiebaSegmenter
        
        segmenter = JiebaSegmenter(strategy="whole")
        tokens = segmenter.segment(sample_chinese_text)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)
    
    def test_jieba_segment_line(self, jieba_available):
        """Test line-by-line segmentation with Jieba."""
        if not jieba_available:
            pytest.skip("jieba not installed")
        
        from qhchina.preprocessing.segmentation import JiebaSegmenter
        
        text = "第一行文本\n第二行文本\n第三行文本"
        
        segmenter = JiebaSegmenter(strategy="line")
        result = segmenter.segment(text)
        
        assert isinstance(result, list)
        assert len(result) == 3  # Three lines
        assert all(isinstance(line, list) for line in result)
    
    def test_jieba_segment_sentence(self, jieba_available):
        """Test sentence segmentation with Jieba."""
        if not jieba_available:
            pytest.skip("jieba not installed")
        
        from qhchina.preprocessing.segmentation import JiebaSegmenter
        
        text = "这是第一句。这是第二句！这是第三句？"
        
        segmenter = JiebaSegmenter(strategy="sentence")
        result = segmenter.segment(text)
        
        assert isinstance(result, list)
        assert len(result) >= 2  # At least 2 sentences
    
    def test_jieba_with_stopwords(self, jieba_available, sample_chinese_text, sample_stopwords):
        """Test segmentation with stopword filtering."""
        if not jieba_available:
            pytest.skip("jieba not installed")
        
        from qhchina.preprocessing.segmentation import JiebaSegmenter
        
        segmenter = JiebaSegmenter(
            strategy="whole",
            filters={'stopwords': sample_stopwords}
        )
        tokens = segmenter.segment(sample_chinese_text)
        
        # Stopwords should be filtered out
        for stopword in sample_stopwords:
            assert stopword not in tokens
    
    def test_jieba_min_word_length(self, jieba_available, sample_chinese_text):
        """Test segmentation with minimum word length filter."""
        if not jieba_available:
            pytest.skip("jieba not installed")
        
        from qhchina.preprocessing.segmentation import JiebaSegmenter
        
        segmenter = JiebaSegmenter(
            strategy="whole",
            filters={'min_word_length': 2}
        )
        tokens = segmenter.segment(sample_chinese_text)
        
        # All tokens should have length >= 2
        assert all(len(t) >= 2 for t in tokens)
    
    def test_jieba_user_dict_list_str(self, jieba_available):
        """Test Jieba with user dictionary as list of strings."""
        if not jieba_available:
            pytest.skip("jieba not installed")
        
        from qhchina.preprocessing.segmentation import JiebaSegmenter
        
        text = "深度学习是人工智能的重要分支"
        
        # With user dict, compound words should be kept together
        segmenter = JiebaSegmenter(user_dict=["深度学习", "人工智能"])
        tokens = segmenter.segment(text)
        
        assert "深度学习" in tokens
        segmenter.close()
    
    def test_jieba_user_dict_list_tuple(self, jieba_available):
        """Test Jieba with user dictionary as list of tuples (word, freq, pos)."""
        if not jieba_available:
            pytest.skip("jieba not installed")
        
        from qhchina.preprocessing.segmentation import JiebaSegmenter
        
        text = "深度学习是人工智能的重要分支"
        
        # With user dict as tuples
        segmenter = JiebaSegmenter(user_dict=[
            ("深度学习", 1000, "n"),
            ("人工智能", 1000, "n")
        ])
        tokens = segmenter.segment(text)
        
        assert "深度学习" in tokens
        segmenter.close()
    
    def test_jieba_user_dict_file_path(self, jieba_available):
        """Test Jieba with user dictionary as file path."""
        if not jieba_available:
            pytest.skip("jieba not installed")
        
        from qhchina.preprocessing.segmentation import JiebaSegmenter
        
        text = "深度学习是人工智能的重要分支"
        
        # Create temp dict file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("深度学习 1000 n\n")
            f.write("人工智能 1000 n\n")
            temp_path = f.name
        
        try:
            segmenter = JiebaSegmenter(user_dict=temp_path)
            tokens = segmenter.segment(text)
            
            assert "深度学习" in tokens
            segmenter.close()
        finally:
            os.unlink(temp_path)
    
    def test_jieba_reset_user_dict(self, jieba_available):
        """Test Jieba reset_user_dict restores default behavior."""
        if not jieba_available:
            pytest.skip("jieba not installed")
        
        from qhchina.preprocessing.segmentation import JiebaSegmenter
        
        text = "深度学习是人工智能的重要分支"
        
        # Create segmenter with user dict
        segmenter = JiebaSegmenter(user_dict=["深度学习"])
        tokens_before = segmenter.segment(text)
        
        # Reset user dict
        segmenter.reset_user_dict()
        tokens_after = segmenter.segment(text)
        
        # Before reset: compound word together
        assert "深度学习" in tokens_before
        # After reset: compound word should be split (default behavior)
        assert "深度学习" not in tokens_after
        
        segmenter.close()
    
    def test_jieba_context_manager(self, jieba_available):
        """Test Jieba with context manager for proper cleanup."""
        if not jieba_available:
            pytest.skip("jieba not installed")
        
        from qhchina.preprocessing.segmentation import JiebaSegmenter
        
        text = "深度学习是人工智能的重要分支"
        
        # Reset Jieba state first to ensure clean test
        temp_segmenter = JiebaSegmenter()
        temp_segmenter.reset_user_dict()
        temp_segmenter.close()
        
        # Test that context manager works with user_dict
        with JiebaSegmenter(user_dict=["深度学习"]) as segmenter:
            tokens = segmenter.segment(text)
            assert "深度学习" in tokens
    
    def test_jieba_empty_text(self, jieba_available):
        """Test Jieba with empty text."""
        if not jieba_available:
            pytest.skip("jieba not installed")
        
        from qhchina.preprocessing.segmentation import JiebaSegmenter
        
        segmenter = JiebaSegmenter(strategy="whole")
        tokens = segmenter.segment("")
        
        assert tokens == []
        segmenter.close()
    
    def test_jieba_whitespace_only(self, jieba_available):
        """Test Jieba with whitespace-only text."""
        if not jieba_available:
            pytest.skip("jieba not installed")
        
        from qhchina.preprocessing.segmentation import JiebaSegmenter
        
        segmenter = JiebaSegmenter(strategy="whole")
        tokens = segmenter.segment("   \t\n   ")
        
        # Should return empty or only whitespace tokens
        assert all(t.strip() == "" for t in tokens) or tokens == []
        segmenter.close()
    
    def test_jieba_chunk_strategy(self, jieba_available):
        """Test Jieba with chunk strategy."""
        if not jieba_available:
            pytest.skip("jieba not installed")
        
        from qhchina.preprocessing.segmentation import JiebaSegmenter
        
        # Long text that will be chunked
        text = "我在年青时候也曾经做过许多梦，后来大半忘却了。" * 50
        
        segmenter = JiebaSegmenter(strategy="chunk", chunk_size=100)
        result = segmenter.segment(text)
        
        # Should return list of lists (chunks)
        assert isinstance(result, list)
        segmenter.close()
    
    def test_jieba_excluded_pos(self, jieba_available):
        """Test Jieba with excluded_pos filter."""
        if not jieba_available:
            pytest.skip("jieba not installed")
        
        from qhchina.preprocessing.segmentation import JiebaSegmenter
        
        text = "我喜欢学习编程"
        
        # Exclude pronouns ('r')
        segmenter = JiebaSegmenter(
            strategy="whole",
            filters={'excluded_pos': {'r'}}  # 'r' is pronoun in Jieba
        )
        tokens = segmenter.segment(text)
        
        # "我" should be filtered out if pos tagging identifies it as pronoun
        # Note: exact behavior depends on Jieba's pos tagging
        assert isinstance(tokens, list)
        segmenter.close()
    
    def test_jieba_combined_filters(self, jieba_available, sample_stopwords):
        """Test Jieba with multiple filters combined."""
        if not jieba_available:
            pytest.skip("jieba not installed")
        
        from qhchina.preprocessing.segmentation import JiebaSegmenter
        
        text = "我在年青时候也曾经做过许多梦"
        
        segmenter = JiebaSegmenter(
            strategy="whole",
            filters={
                'stopwords': sample_stopwords,
                'min_word_length': 2
            }
        )
        tokens = segmenter.segment(text)
        
        # Stopwords should be filtered
        for sw in sample_stopwords:
            assert sw not in tokens
        
        # All tokens should have length >= 2
        assert all(len(t) >= 2 for t in tokens)
        segmenter.close()


class TestPKUSegmenter:
    """Tests for PKUSeg-based segmentation (if available)."""
    
    @pytest.fixture
    def pkuseg_available(self):
        """Check if pkuseg is available."""
        try:
            import pkuseg
            return True
        except ImportError:
            return False
    
    def test_pkuseg_segment_whole(self, pkuseg_available):
        """Test whole-text segmentation with PKUSeg."""
        if not pkuseg_available:
            pytest.skip("pkuseg not installed")
        
        from qhchina.preprocessing.segmentation import PKUSegmenter
        
        text = "我在年青时候也曾经做过许多梦"
        
        segmenter = PKUSegmenter(strategy="whole")
        tokens = segmenter.segment(text)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)
        segmenter.close()
    
    def test_pkuseg_user_dict_list_str(self, pkuseg_available):
        """Test PKUSeg with user dictionary as list of strings."""
        if not pkuseg_available:
            pytest.skip("pkuseg not installed")
        
        from qhchina.preprocessing.segmentation import PKUSegmenter
        
        text = "深度学习是人工智能的重要分支"
        
        # With user dict, compound words should be kept together
        segmenter = PKUSegmenter(user_dict=["深度学习", "人工智能"])
        tokens = segmenter.segment(text)
        
        assert "深度学习" in tokens
        segmenter.close()
    
    def test_pkuseg_user_dict_file_path(self, pkuseg_available):
        """Test PKUSeg with user dictionary as file path."""
        if not pkuseg_available:
            pytest.skip("pkuseg not installed")
        
        from qhchina.preprocessing.segmentation import PKUSegmenter
        
        text = "深度学习是人工智能的重要分支"
        
        # Create temp dict file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("深度学习\n")
            f.write("人工智能\n")
            temp_path = f.name
        
        try:
            segmenter = PKUSegmenter(user_dict=temp_path)
            tokens = segmenter.segment(text)
            
            assert "深度学习" in tokens
            segmenter.close()
        finally:
            os.unlink(temp_path)
    
    def test_pkuseg_reset_user_dict(self, pkuseg_available):
        """Test PKUSeg reset_user_dict restores default behavior."""
        if not pkuseg_available:
            pytest.skip("pkuseg not installed")
        
        from qhchina.preprocessing.segmentation import PKUSegmenter
        
        text = "深度学习是人工智能的重要分支"
        
        # Create segmenter with user dict
        segmenter = PKUSegmenter(user_dict=["深度学习"])
        tokens_before = segmenter.segment(text)
        
        # Reset user dict
        segmenter.reset_user_dict()
        tokens_after = segmenter.segment(text)
        
        # Before reset: compound word together
        assert "深度学习" in tokens_before
        # After reset: compound word should be split (default behavior)
        assert "深度学习" not in tokens_after
        
        segmenter.close()
    
    def test_pkuseg_pos_tagging(self, pkuseg_available):
        """Test PKUSeg with POS tagging enabled."""
        if not pkuseg_available:
            pytest.skip("pkuseg not installed")
        
        from qhchina.preprocessing.segmentation import PKUSegmenter
        
        text = "我在年青时候也曾经做过许多梦"
        
        # POS tagging returns words only (tags are used internally for filtering)
        segmenter = PKUSegmenter(pos_tagging=True)
        tokens = segmenter.segment(text)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)
        segmenter.close()
    
    def test_pkuseg_context_manager(self, pkuseg_available):
        """Test PKUSeg with context manager for proper cleanup."""
        if not pkuseg_available:
            pytest.skip("pkuseg not installed")
        
        from qhchina.preprocessing.segmentation import PKUSegmenter
        
        text = "深度学习是人工智能的重要分支"
        
        with PKUSegmenter(user_dict=["深度学习"]) as segmenter:
            tokens = segmenter.segment(text)
            assert "深度学习" in tokens
    
    def test_pkuseg_empty_text(self, pkuseg_available):
        """Test PKUSeg with empty text."""
        if not pkuseg_available:
            pytest.skip("pkuseg not installed")
        
        from qhchina.preprocessing.segmentation import PKUSegmenter
        
        segmenter = PKUSegmenter(strategy="whole")
        tokens = segmenter.segment("")
        
        assert tokens == []
        segmenter.close()
    
    def test_pkuseg_sentence_strategy(self, pkuseg_available):
        """Test PKUSeg with sentence strategy."""
        if not pkuseg_available:
            pytest.skip("pkuseg not installed")
        
        from qhchina.preprocessing.segmentation import PKUSegmenter
        
        text = "这是第一句。这是第二句！这是第三句？"
        
        segmenter = PKUSegmenter(strategy="sentence")
        result = segmenter.segment(text)
        
        assert isinstance(result, list)
        assert len(result) >= 2
        segmenter.close()


class TestCreateSegmenter:
    """Tests for the create_segmenter factory function."""
    
    def test_create_jieba_segmenter(self):
        """Test creating Jieba segmenter via factory."""
        try:
            import jieba
        except ImportError:
            pytest.skip("jieba not installed")
        
        from qhchina.preprocessing.segmentation import create_segmenter
        
        segmenter = create_segmenter(backend="jieba")
        assert segmenter is not None
        segmenter.close()
    
    def test_create_pkuseg_segmenter(self):
        """Test creating PKUSeg segmenter via factory."""
        try:
            import pkuseg
        except ImportError:
            pytest.skip("pkuseg not installed")
        
        from qhchina.preprocessing.segmentation import create_segmenter
        
        segmenter = create_segmenter(backend="pkuseg")
        assert segmenter is not None
        segmenter.close()
    
    def test_create_segmenter_with_user_dict(self):
        """Test creating segmenter with user_dict via factory."""
        try:
            import jieba
        except ImportError:
            pytest.skip("jieba not installed")
        
        from qhchina.preprocessing.segmentation import create_segmenter, JiebaSegmenter
        
        text = "深度学习是人工智能的重要分支"
        
        # First, reset any previous Jieba state to ensure clean test
        temp_segmenter = JiebaSegmenter()
        temp_segmenter.reset_user_dict()
        temp_segmenter.close()
        
        # Now test that user_dict works via factory
        segmenter = create_segmenter(
            backend="jieba",
            user_dict=["深度学习", "人工智能"]
        )
        tokens = segmenter.segment(text)
        
        assert "深度学习" in tokens
        segmenter.close()
    
    def test_create_segmenter_invalid_backend(self):
        """Test that invalid backend raises error."""
        from qhchina.preprocessing.segmentation import create_segmenter
        
        with pytest.raises(ValueError):
            create_segmenter(backend="invalid_backend")
    
    def test_create_segmenter_with_filters(self):
        """Test creating segmenter with filters via factory."""
        try:
            import jieba
        except ImportError:
            pytest.skip("jieba not installed")
        
        from qhchina.preprocessing.segmentation import create_segmenter
        
        stopwords = {"的", "了", "是"}
        
        segmenter = create_segmenter(
            backend="jieba",
            filters={
                'stopwords': stopwords,
                'min_word_length': 2
            }
        )
        
        tokens = segmenter.segment("我喜欢学习的过程")
        
        # Stopwords should be filtered
        for sw in stopwords:
            assert sw not in tokens
        
        segmenter.close()
    
    def test_create_segmenter_with_strategy(self):
        """Test creating segmenter with different strategies."""
        try:
            import jieba
        except ImportError:
            pytest.skip("jieba not installed")
        
        from qhchina.preprocessing.segmentation import create_segmenter
        
        for strategy in ["whole", "line", "sentence"]:
            segmenter = create_segmenter(backend="jieba", strategy=strategy)
            assert segmenter.strategy == strategy
            segmenter.close()


# =============================================================================
# Edge Cases and Validation
# =============================================================================

class TestSegmentationEdgeCases:
    """Tests for edge cases in segmentation."""
    
    @pytest.fixture
    def jieba_available(self):
        """Check if jieba is available."""
        try:
            import jieba
            return True
        except ImportError:
            return False
    
    def test_unicode_characters(self, jieba_available):
        """Test segmentation with various Unicode characters."""
        if not jieba_available:
            pytest.skip("jieba not installed")
        
        from qhchina.preprocessing.segmentation import JiebaSegmenter
        
        # Text with various Unicode characters
        text = "我爱🎉编程！Hello世界。"
        
        segmenter = JiebaSegmenter(strategy="whole")
        tokens = segmenter.segment(text)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        segmenter.close()
    
    def test_punctuation_handling(self, jieba_available):
        """Test handling of Chinese and English punctuation."""
        if not jieba_available:
            pytest.skip("jieba not installed")
        
        from qhchina.preprocessing.segmentation import JiebaSegmenter
        
        # Text with mixed punctuation
        text = "你好！Hello, 世界。How are you?"
        
        segmenter = JiebaSegmenter(strategy="whole")
        tokens = segmenter.segment(text)
        
        assert isinstance(tokens, list)
        segmenter.close()
    
    def test_numbers_in_text(self, jieba_available):
        """Test handling of numbers in text."""
        if not jieba_available:
            pytest.skip("jieba not installed")
        
        from qhchina.preprocessing.segmentation import JiebaSegmenter
        
        text = "今天是2024年1月1日，温度是25.5度。"
        
        segmenter = JiebaSegmenter(strategy="whole")
        tokens = segmenter.segment(text)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        segmenter.close()
    
    def test_classical_chinese(self, jieba_available):
        """Test segmentation of classical Chinese text."""
        if not jieba_available:
            pytest.skip("jieba not installed")
        
        from qhchina.preprocessing.segmentation import JiebaSegmenter
        
        # Classical Chinese text
        text = "子曰：學而時習之，不亦說乎？"
        
        segmenter = JiebaSegmenter(strategy="whole")
        tokens = segmenter.segment(text)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        segmenter.close()
    
    def test_mixed_script_text(self, jieba_available):
        """Test handling of mixed Chinese/English text."""
        if not jieba_available:
            pytest.skip("jieba not installed")
        
        from qhchina.preprocessing.segmentation import JiebaSegmenter
        
        text = "Python是最好的programming语言之一"
        
        segmenter = JiebaSegmenter(strategy="whole")
        tokens = segmenter.segment(text)
        
        assert isinstance(tokens, list)
        assert "Python" in tokens or "python" in [t.lower() for t in tokens]
        segmenter.close()
    
    def test_very_long_text(self, jieba_available):
        """Test segmentation of very long text."""
        if not jieba_available:
            pytest.skip("jieba not installed")
        
        from qhchina.preprocessing.segmentation import JiebaSegmenter
        
        # Create a long text
        text = "我喜欢学习编程。" * 1000
        
        segmenter = JiebaSegmenter(strategy="whole")
        tokens = segmenter.segment(text)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        segmenter.close()
    
    def test_single_character_text(self, jieba_available):
        """Test segmentation of single character."""
        if not jieba_available:
            pytest.skip("jieba not installed")
        
        from qhchina.preprocessing.segmentation import JiebaSegmenter
        
        segmenter = JiebaSegmenter(strategy="whole")
        tokens = segmenter.segment("我")
        
        assert isinstance(tokens, list)
        segmenter.close()


class TestSpacySegmenter:
    """Tests for spaCy-based segmentation (if available)."""
    
    @pytest.fixture
    def spacy_available(self):
        """Check if spaCy with Chinese model is available."""
        try:
            import spacy
            nlp = spacy.load("zh_core_web_sm")
            return True
        except (ImportError, OSError):
            return False
    
    def test_spacy_segment_whole(self, spacy_available, sample_chinese_text):
        """Test whole-text segmentation with spaCy."""
        if not spacy_available:
            pytest.skip("spaCy with Chinese model not installed")
        
        from qhchina.preprocessing.segmentation import SpacySegmenter
        
        segmenter = SpacySegmenter(strategy="whole")
        tokens = segmenter.segment(sample_chinese_text)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        segmenter.close()
    
    def test_spacy_user_dict_list_str(self, spacy_available):
        """Test spaCy with user dictionary as list of strings."""
        if not spacy_available:
            pytest.skip("spaCy with Chinese model not installed")
        
        from qhchina.preprocessing.segmentation import SpacySegmenter
        
        text = "深度学习是人工智能的重要分支"
        
        # Note: spaCy user dict only works if the tokenizer supports pkuseg_update_user_dict
        segmenter = SpacySegmenter(user_dict=["深度学习", "人工智能"])
        tokens = segmenter.segment(text)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        segmenter.close()
    
    def test_spacy_context_manager(self, spacy_available, sample_chinese_text):
        """Test spaCy with context manager for proper cleanup."""
        if not spacy_available:
            pytest.skip("spaCy with Chinese model not installed")
        
        from qhchina.preprocessing.segmentation import SpacySegmenter
        
        with SpacySegmenter() as segmenter:
            tokens = segmenter.segment(sample_chinese_text)
            assert isinstance(tokens, list)
            assert len(tokens) > 0


class TestBertSegmenter:
    """Tests for BERT-based segmentation (if available)."""
    
    @pytest.fixture
    def bert_available(self):
        """Check if transformers is available."""
        try:
            import transformers
            return True
        except ImportError:
            return False
    
    def test_bert_init(self, bert_available):
        """Test BertSegmenter initialization."""
        if not bert_available:
            pytest.skip("transformers not installed")
        
        from qhchina.preprocessing.segmentation import BertSegmenter
        
        # Just test that it can be imported, actual segmentation may require model download
        assert BertSegmenter is not None
    
    def test_bert_user_dict_warning(self, bert_available, caplog):
        """Test that BertSegmenter logs warning when user_dict is provided."""
        if not bert_available:
            pytest.skip("transformers not installed")
        
        import logging
        from qhchina.preprocessing.segmentation import BertSegmenter
        
        # This will fail to create due to missing model, but the warning should be logged
        # before the model loading fails
        try:
            with caplog.at_level(logging.WARNING):
                segmenter = BertSegmenter(
                    model_name="bert-base-chinese",
                    user_dict=["深度学习"]
                )
                segmenter.close()
        except Exception:
            pass  # Expected to fail due to missing model
        
        # Check that warning was logged
        assert any("user_dict is not supported for BertSegmenter" in record.message 
                   for record in caplog.records)


class TestLLMSegmenter:
    """Tests for LLM-based segmentation (if available)."""
    
    @pytest.fixture
    def openai_available(self):
        """Check if openai is available."""
        try:
            import openai
            return True
        except ImportError:
            return False
    
    def test_llm_user_dict_warning(self, openai_available, caplog):
        """Test that LLMSegmenter logs warning when user_dict is provided."""
        if not openai_available:
            pytest.skip("openai not installed")
        
        import logging
        from qhchina.preprocessing.segmentation import LLMSegmenter
        
        with caplog.at_level(logging.WARNING):
            # Create with fake credentials - warning should be logged during init
            segmenter = LLMSegmenter(
                api_key="fake-key",
                model="gpt-4",
                endpoint="https://api.openai.com/v1",
                user_dict=["深度学习"]
            )
            segmenter.close()
        
        # Check that warning was logged
        assert any("user_dict is not supported for LLMSegmenter" in record.message 
                   for record in caplog.records)
