"""
Tests for qhchina.helpers module (texts, fonts, stopwords).
"""
import pytest
from pathlib import Path


# =============================================================================
# Load Text Tests
# =============================================================================

class TestLoadText:
    """Tests for load_text and load_texts functions."""
    
    def test_load_text_basic(self, texts_dir):
        """Test loading a single text file."""
        from qhchina.helpers import load_text
        
        text_file = texts_dir / "民国_鲁迅_呐喊_1923.txt"
        text = load_text(str(text_file))
        
        assert isinstance(text, str)
        assert len(text) > 0
        assert "我" in text  # Basic Chinese character check
    
    def test_load_texts_multiple(self, texts_dir):
        """Test loading multiple text files."""
        from qhchina.helpers import load_texts
        
        files = [
            str(texts_dir / "民国_鲁迅_呐喊_1923.txt"),
            str(texts_dir / "民国_鲁迅_彷徨_1926.txt"),
        ]
        texts = load_texts(files)
        
        assert isinstance(texts, list)
        assert len(texts) == 2
        assert all(isinstance(t, str) for t in texts)
    
    def test_load_text_invalid_path(self):
        """Test that loading non-existent file raises error."""
        from qhchina.helpers import load_text
        
        with pytest.raises(FileNotFoundError):
            load_text("/nonexistent/path/file.txt")
    
    def test_load_text_invalid_filename_type(self):
        """Test that non-string filename raises ValueError."""
        from qhchina.helpers import load_text
        
        with pytest.raises(ValueError, match="filename must be a string"):
            load_text(123)
    
    def test_load_text_auto_encoding(self, texts_dir):
        """Test loading text with automatic encoding detection."""
        from qhchina.helpers import load_text
        
        text_file = texts_dir / "民国_鲁迅_呐喊_1923.txt"
        text = load_text(str(text_file), encoding="auto")
        
        assert isinstance(text, str)
        assert len(text) > 0
    
    def test_load_texts_single_string_input(self, texts_dir):
        """Test that load_texts accepts a single string path."""
        from qhchina.helpers import load_texts
        
        text_file = str(texts_dir / "民国_鲁迅_呐喊_1923.txt")
        texts = load_texts(text_file)
        
        assert isinstance(texts, list)
        assert len(texts) == 1


class TestLoadStopwords:
    """Tests for stopword loading functions."""
    
    def test_load_stopwords_simplified(self):
        """Test loading simplified Chinese stopwords."""
        from qhchina.helpers import load_stopwords
        
        stopwords = load_stopwords("zh_sim")
        
        assert isinstance(stopwords, set)
        assert len(stopwords) > 0
        assert "的" in stopwords  # Common stopword
    
    def test_load_stopwords_traditional(self):
        """Test loading traditional Chinese stopwords."""
        from qhchina.helpers import load_stopwords
        
        stopwords = load_stopwords("zh_tr")
        
        assert isinstance(stopwords, set)
        assert len(stopwords) > 0
    
    def test_load_stopwords_classical(self):
        """Test loading classical Chinese stopwords."""
        from qhchina.helpers import load_stopwords
        
        stopwords = load_stopwords("zh_cl_sim")
        
        assert isinstance(stopwords, set)
        assert len(stopwords) > 0
    
    def test_load_stopwords_invalid_language(self):
        """Test that invalid language raises ValueError."""
        from qhchina.helpers import load_stopwords
        
        with pytest.raises(ValueError, match="not found"):
            load_stopwords("invalid_language_xyz")
    
    def test_load_stopwords_all_available(self):
        """Test loading all available stopword languages."""
        from qhchina.helpers import load_stopwords, get_stopword_languages
        
        languages = get_stopword_languages()
        
        for lang in languages:
            stopwords = load_stopwords(lang)
            assert isinstance(stopwords, set)
            assert len(stopwords) > 0
    
    def test_get_stopword_languages(self):
        """Test listing available stopword languages."""
        from qhchina.helpers import get_stopword_languages
        
        languages = get_stopword_languages()
        
        assert isinstance(languages, list)
        assert "zh_sim" in languages
        assert "zh_tr" in languages


class TestSplitIntoChunks:
    """Tests for text chunking function."""
    
    def test_split_into_chunks_basic(self, sample_chinese_text):
        """Test basic text chunking."""
        from qhchina.helpers import split_into_chunks
        
        chunks = split_into_chunks(sample_chinese_text, chunk_size=10)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(c, str) for c in chunks)
    
    def test_split_into_chunks_overlap(self, sample_chinese_text):
        """Test chunking with overlap (float between 0-1)."""
        from qhchina.helpers import split_into_chunks
        
        # overlap=0.3 means 30% overlap
        chunks = split_into_chunks(sample_chinese_text, chunk_size=10, overlap=0.3)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
    
    def test_split_into_chunks_list_input(self, sample_tokenized):
        """Test chunking with list input (tokens)."""
        from qhchina.helpers import split_into_chunks
        
        chunks = split_into_chunks(sample_tokenized, chunk_size=5)
        
        assert isinstance(chunks, list)
        assert all(isinstance(c, list) for c in chunks)
    
    def test_split_into_chunks_empty_input(self):
        """Test chunking with empty input returns empty list."""
        from qhchina.helpers import split_into_chunks
        
        assert split_into_chunks("", chunk_size=5) == []
        assert split_into_chunks([], chunk_size=5) == []
    
    def test_split_into_chunks_invalid_chunk_size_zero(self):
        """Test that chunk_size=0 raises ValueError."""
        from qhchina.helpers import split_into_chunks
        
        with pytest.raises(ValueError, match="chunk_size must be greater than 0"):
            split_into_chunks("test", chunk_size=0)
    
    def test_split_into_chunks_invalid_chunk_size_negative(self):
        """Test that negative chunk_size raises ValueError."""
        from qhchina.helpers import split_into_chunks
        
        with pytest.raises(ValueError, match="chunk_size must be greater than 0"):
            split_into_chunks("test", chunk_size=-1)
    
    def test_split_into_chunks_invalid_overlap_one(self):
        """Test that overlap=1.0 raises ValueError."""
        from qhchina.helpers import split_into_chunks
        
        with pytest.raises(ValueError, match="Overlap must be between 0 and 1"):
            split_into_chunks("test text", chunk_size=2, overlap=1.0)
    
    def test_split_into_chunks_invalid_overlap_negative(self):
        """Test that negative overlap raises ValueError."""
        from qhchina.helpers import split_into_chunks
        
        with pytest.raises(ValueError, match="Overlap must be between 0 and 1"):
            split_into_chunks("test text", chunk_size=2, overlap=-0.1)
    
    def test_split_into_chunks_sequence_shorter_than_chunk(self):
        """Test when sequence is shorter than chunk_size."""
        from qhchina.helpers import split_into_chunks
        
        text = "short"
        chunks = split_into_chunks(text, chunk_size=100)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_split_into_chunks_exact_division(self):
        """Test when sequence divides exactly into chunks."""
        from qhchina.helpers import split_into_chunks
        
        text = "abcdefghij"  # 10 chars
        chunks = split_into_chunks(text, chunk_size=5, overlap=0.0)
        
        assert len(chunks) == 2
        assert chunks[0] == "abcde"
        assert chunks[1] == "fghij"
    
    def test_split_into_chunks_with_high_overlap(self, sample_chinese_text):
        """Test chunking with high overlap (0.9)."""
        from qhchina.helpers import split_into_chunks
        
        chunks = split_into_chunks(sample_chinese_text, chunk_size=10, overlap=0.9)
        
        # With 90% overlap, should have many overlapping chunks
        assert isinstance(chunks, list)
        assert len(chunks) > 1


class TestDetectEncoding:
    """Tests for encoding detection function."""
    
    def test_detect_encoding_utf8(self, texts_dir):
        """Test detecting UTF-8 encoding."""
        from qhchina.helpers import detect_encoding
        
        # The test files should be UTF-8 encoded
        text_file = texts_dir / "民国_鲁迅_呐喊_1923.txt"
        encoding = detect_encoding(str(text_file))
        
        # Should detect as UTF-8 or a compatible encoding
        assert encoding.lower() in ['utf-8', 'utf8', 'ascii']
    
    def test_detect_encoding_with_num_bytes(self, texts_dir):
        """Test encoding detection with custom num_bytes."""
        from qhchina.helpers import detect_encoding
        
        text_file = texts_dir / "红楼梦.txt"
        encoding = detect_encoding(str(text_file), num_bytes=5000)
        
        assert isinstance(encoding, str)
        assert len(encoding) > 0
    
    def test_detect_encoding_returns_string(self, texts_dir):
        """Test that detect_encoding always returns a string."""
        from qhchina.helpers import detect_encoding
        
        text_file = texts_dir / "宋史.txt"
        encoding = detect_encoding(str(text_file))
        
        assert isinstance(encoding, str)


class TestFonts:
    """Tests for font management functions."""
    
    def test_list_available_fonts(self):
        """Test listing available fonts (returns dict)."""
        from qhchina.helpers import list_available_fonts
        
        fonts = list_available_fonts()
        
        assert isinstance(fonts, dict)
        assert len(fonts) > 0
    
    def test_get_font_path(self):
        """Test getting font path."""
        from qhchina.helpers import get_font_path
        
        # Should return a path for one of the built-in fonts
        path = get_font_path()
        
        assert path is not None
        assert Path(path).exists()
    
    def test_get_font_path_with_alias(self):
        """Test getting font path with alias."""
        from qhchina.helpers import get_font_path
        
        # Test various aliases
        for alias in ['sans', 'sans-tc', 'serif-tc', 'serif-sc']:
            path = get_font_path(alias)
            assert path is not None
            assert Path(path).exists()
    
    def test_get_font_path_invalid_font(self):
        """Test that invalid font name raises ValueError."""
        from qhchina.helpers import get_font_path
        
        with pytest.raises(ValueError, match="Unknown font"):
            get_font_path("NonExistentFont")
    
    def test_load_fonts(self):
        """Test loading fonts for matplotlib."""
        from qhchina.helpers import load_fonts
        
        # Should not raise
        load_fonts()
    
    def test_load_fonts_verbose(self):
        """Test loading fonts with verbose mode returns font info."""
        from qhchina.helpers import load_fonts
        
        result = load_fonts(verbose=True)
        
        assert isinstance(result, list)
        if len(result) > 0:
            assert 'font_name' in result[0]
            assert 'aliases' in result[0]
            assert 'path' in result[0]
    
    def test_load_fonts_with_target(self):
        """Test loading fonts with specific target font."""
        from qhchina.helpers import load_fonts
        
        # Should not raise
        load_fonts(target_font='sans')
        load_fonts(target_font='serif-tc')
    
    def test_current_font(self):
        """Test getting current font."""
        from qhchina.helpers import current_font, load_fonts
        
        load_fonts()
        font = current_font()
        
        assert font is not None
    
    def test_set_font(self):
        """Test setting font."""
        from qhchina.helpers.fonts import set_font, current_font, load_fonts
        
        load_fonts()
        
        # Set to a known font
        set_font('Noto Sans CJK TC')
        
        # Verify it was set
        font = current_font()
        assert font is not None
    
    def test_set_font_with_alias(self):
        """Test setting font with alias."""
        from qhchina.helpers.fonts import set_font, load_fonts
        
        load_fonts()
        
        # Should not raise
        set_font('sans')
        set_font('serif-tc')
    
    def test_list_font_aliases(self):
        """Test listing font aliases."""
        from qhchina.helpers.fonts import list_font_aliases
        
        aliases = list_font_aliases()
        
        assert isinstance(aliases, dict)
        assert 'sans' in aliases
        assert 'serif-tc' in aliases
