"""
Tests for qhchina.preprocessing module.
"""
import pytest


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
