"""
Pytest configuration and shared fixtures for qhchina tests.
"""
import pytest
from pathlib import Path

# Suppress matplotlib display during tests
import matplotlib
matplotlib.use('Agg')


# =============================================================================
# Path Fixtures
# =============================================================================

@pytest.fixture
def tests_dir():
    """Path to the tests directory."""
    return Path(__file__).parent


@pytest.fixture
def texts_dir(tests_dir):
    """Path to the texts directory containing test corpora."""
    return tests_dir / "texts"


# =============================================================================
# Sample Text Fixtures (small data for fast tests)
# =============================================================================

@pytest.fixture
def sample_chinese_text():
    """A small sample of Chinese text for quick tests."""
    return "我在年青时候也曾经做过许多梦，后来大半忘却了，但自己也并不以为可惜。"


@pytest.fixture
def sample_tokenized():
    """Pre-tokenized Chinese text (character-level) for quick tests."""
    text = "我在年青时候也曾经做过许多梦后来大半忘却了但自己也并不以为可惜"
    return list(text)


@pytest.fixture
def sample_documents():
    """Small corpus of tokenized documents for testing."""
    docs = [
        list("我在年青时候也曾经做过许多梦"),
        list("后来大半忘却了但自己也并不以为可惜"),
        list("所谓回忆者虽说可以使人欢欣"),
        list("有时也不免使人寂寞"),
        list("使精神的丝缕还牵着已逝的寂寞的时光"),
    ]
    return docs


@pytest.fixture
def sample_corpus_dict():
    """Small corpus as dict for stylometry tests."""
    return {
        "author_a_1": list("我在年青时候也曾经做过许多梦后来大半忘却了"),
        "author_a_2": list("但自己也并不以为可惜所谓回忆者虽说可以使人欢欣"),
        "author_b_1": list("有时也不免使人寂寞使精神的丝缕还牵着已逝的时光"),
        "author_b_2": list("又有什么意味呢而我偏苦于不能全忘却这不能全忘的"),
    }


@pytest.fixture
def larger_documents(texts_dir):
    """Load a few lines from a real text file for more realistic tests."""
    text_file = texts_dir / "民国_鲁迅_呐喊_1923.txt"
    if not text_file.exists():
        pytest.skip(f"Test file not found: {text_file}")
    
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split into paragraphs and take first 10, tokenize by character
    paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 50][:10]
    return [list(p) for p in paragraphs]


# =============================================================================
# Utility Fixtures
# =============================================================================

@pytest.fixture
def sample_stopwords():
    """Small set of Chinese stopwords for testing."""
    return {"的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "也"}


# =============================================================================
# Historical Text Fixtures (宋史/明史 for temporal analysis tests)
# =============================================================================

@pytest.fixture
def songshi_excerpt(texts_dir):
    """Load an excerpt from 宋史 (Song Dynasty history) for testing."""
    text_file = texts_dir / "宋史.txt"
    if not text_file.exists():
        pytest.skip(f"Test file not found: {text_file}")
    
    with open(text_file, 'r', encoding='utf-8') as f:
        # Read first 50000 characters for testing
        text = f.read(50000)
    
    # Split into sentences and tokenize by character
    sentences = [list(s.strip()) for s in text.split('。') if len(s.strip()) > 10][:100]
    return sentences


@pytest.fixture
def mingshi_excerpt(texts_dir):
    """Load an excerpt from 明史 (Ming Dynasty history) for testing."""
    text_file = texts_dir / "明史.txt"
    if not text_file.exists():
        pytest.skip(f"Test file not found: {text_file}")
    
    with open(text_file, 'r', encoding='utf-8') as f:
        # Read first 50000 characters for testing
        text = f.read(50000)
    
    # Split into sentences and tokenize by character
    sentences = [list(s.strip()) for s in text.split('。') if len(s.strip()) > 10][:100]
    return sentences


@pytest.fixture
def song_ming_corpora(songshi_excerpt, mingshi_excerpt):
    """Combined corpora from 宋史 and 明史 for temporal analysis."""
    return {
        'song': songshi_excerpt,
        'ming': mingshi_excerpt
    }


@pytest.fixture
def song_ming_flat(songshi_excerpt, mingshi_excerpt):
    """Flat token lists from 宋史 and 明史 for corpus comparison."""
    song_tokens = [token for sentence in songshi_excerpt for token in sentence]
    ming_tokens = [token for sentence in mingshi_excerpt for token in sentence]
    return {
        'song': song_tokens,
        'ming': ming_tokens
    }
