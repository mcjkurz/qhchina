"""
Tests for qhchina.helpers module (texts, fonts, stopwords).
"""
import pytest
from pathlib import Path


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
    
    def test_load_fonts(self):
        """Test loading fonts for matplotlib."""
        from qhchina.helpers import load_fonts
        
        # Should not raise
        load_fonts()
    
    def test_current_font(self):
        """Test getting current font."""
        from qhchina.helpers import current_font, load_fonts
        
        load_fonts()
        font = current_font()
        
        assert font is not None
