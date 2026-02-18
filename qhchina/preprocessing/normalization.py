"""
Chinese text normalization module.

Provides functions to standardize text by normalizing:
- Script (Traditional ↔ Simplified Chinese)
- Unicode normalization (NFC, NFD, NFKC, NFKD)
- Whitespace (collapse, strip, remove)
- Punctuation width (full-width ↔ half-width)
- Number and letter width
- Quotation mark styles

Usage:
    from qhchina.preprocessing.normalization import normalize
    
    # Basic normalization
    text = normalize(raw_text, {'unicode': 'NFC', 'whitespace': 'collapse'})
    
    # Script conversion (requires: pip install opencc)
    text = normalize(raw_text, {'conversion': 't2s'})  # Traditional → Simplified
    
    # Full normalization
    text = normalize(raw_text, {
        'conversion': 't2tw',  # Traditional → Taiwan standard (normalizes variants)
        'unicode': 'NFC',
        'whitespace': 'collapse',
        'punctuation': 'full',
        'numbers': 'half',
        'quotes': 'smart',
    })
"""

import re
import unicodedata
from functools import lru_cache
from typing import Literal, TypedDict

__all__ = [
    'normalize',
    'NormalizeOptions',
]


class NormalizeOptions(TypedDict, total=False):
    """
    Normalization options dictionary.
    
    All keys are optional - only specified options are applied.
    
    Keys:
        conversion: Script/variant conversion using OpenCC. Values:
            - 't2s': Traditional → Simplified
            - 's2t': Simplified → Traditional
            - 't2tw': Traditional → Taiwan standard (normalizes variants)
            - 't2hk': Traditional → Hong Kong standard
            - 's2tw': Simplified → Taiwan traditional
            - 's2hk': Simplified → Hong Kong traditional
            - Any valid OpenCC configuration name
        
        unicode: Unicode normalization form.
            - 'NFC': Canonical composition (recommended)
            - 'NFD': Canonical decomposition
            - 'NFKC': Compatibility composition
            - 'NFKD': Compatibility decomposition
        
        whitespace: Whitespace handling.
            - 'collapse': Collapse multiple spaces/tabs/newlines to single, strip text
            - 'strip': Strip leading/trailing whitespace from each line
            - 'remove': Remove all whitespace (spaces, tabs, newlines)
        
        punctuation: Punctuation width.
            - 'full': Convert to full-width (，。！？)
            - 'half': Convert to half-width (,.!?)
        
        numbers: Digit width.
            - 'full': Convert to full-width (０-９)
            - 'half': Convert to half-width (0-9)
        
        letters: Letter width.
            - 'full': Convert to full-width (Ａ-Ｚ)
            - 'half': Convert to half-width (A-Z)
        
        quotes: Quotation mark style (smart nesting).
            - 'straight': ASCII quotes (" ')
            - 'smart': Typographic curly quotes (" " ' ')
            - 'corner': East Asian corner brackets (「 」 『 』)
            - 'guillemets': French-style angle quotes (« » ‹ ›)
    """
    conversion: str
    unicode: Literal['NFC', 'NFD', 'NFKC', 'NFKD']
    whitespace: Literal['collapse', 'strip', 'remove']
    punctuation: Literal['full', 'half']
    numbers: Literal['full', 'half']
    letters: Literal['full', 'half']
    quotes: Literal['straight', 'smart', 'corner', 'guillemets']


# =============================================================================
# Pre-compiled mappings for efficiency (computed once at module load)
# =============================================================================

# Punctuation-only width conversion tables
# Maps common punctuation between half-width (ASCII) and full-width (CJK) forms
_PUNCT_HALF = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
_PUNCT_FULL = '！＂＃＄％＆＇（）＊＋，－．／：；＜＝＞？＠［＼］＾＿｀｛｜｝～'
_TRANS_PUNCT_TO_FULL = str.maketrans(_PUNCT_HALF, _PUNCT_FULL)
# For half-width conversion, also map ideographic space (U+3000) to regular space
_TRANS_PUNCT_TO_HALF = str.maketrans(_PUNCT_FULL + '\u3000', _PUNCT_HALF + ' ')

# Digits: pre-built translation tables
_DIGITS_FULL = '０１２３４５６７８９'
_DIGITS_HALF = '0123456789'
_TRANS_DIGITS_TO_HALF = str.maketrans(_DIGITS_FULL, _DIGITS_HALF)
_TRANS_DIGITS_TO_FULL = str.maketrans(_DIGITS_HALF, _DIGITS_FULL)

# Letters: pre-built translation tables
_LETTERS_FULL_UPPER = 'ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ'
_LETTERS_FULL_LOWER = 'ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ'
_LETTERS_HALF_UPPER = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
_LETTERS_HALF_LOWER = 'abcdefghijklmnopqrstuvwxyz'
_LETTERS_FULL = _LETTERS_FULL_UPPER + _LETTERS_FULL_LOWER
_LETTERS_HALF = _LETTERS_HALF_UPPER + _LETTERS_HALF_LOWER
_TRANS_LETTERS_TO_HALF = str.maketrans(_LETTERS_FULL, _LETTERS_HALF)
_TRANS_LETTERS_TO_FULL = str.maketrans(_LETTERS_HALF, _LETTERS_FULL)

# Quote styles: (primary_open, primary_close, nested_open, nested_close)
# Using Unicode escapes to avoid encoding/parsing issues
_QUOTE_STYLES = {
    'straight':   ('"', '"', "'", "'"),
    'smart':      ('\u201C', '\u201D', '\u2018', '\u2019'),  # " " ' '
    'corner':     ('\u300C', '\u300D', '\u300E', '\u300F'),  # 「 」 『 』
    'guillemets': ('\u00AB', '\u00BB', '\u2039', '\u203A'),  # « » ‹ ›
}

# All quote characters for detection (opening and closing variants)
# Using Unicode escapes to avoid encoding/parsing issues
_OPENING_QUOTES = {
    '"',        # ASCII double quote (ambiguous)
    "'",        # ASCII single quote (ambiguous)
    '\uFF02',   # Fullwidth double quote ＂ (ambiguous)
    '\uFF07',   # Fullwidth single quote ＇ (ambiguous)
    '\u300C',   # CJK corner bracket 「
    '\u300E',   # CJK white corner bracket 『
    '\u201C',   # Left double quotation mark "
    '\u2018',   # Left single quotation mark '
    '\u2039',   # Single left-pointing angle quotation mark ‹
    '\u00AB',   # Left-pointing double angle quotation mark «
    '\uFF62',   # Halfwidth left corner bracket ｢
}
_CLOSING_QUOTES = {
    '"',        # ASCII double quote (ambiguous)
    "'",        # ASCII single quote (ambiguous)
    '\uFF02',   # Fullwidth double quote ＂ (ambiguous)
    '\uFF07',   # Fullwidth single quote ＇ (ambiguous)
    '\u300D',   # CJK corner bracket 」
    '\u300F',   # CJK white corner bracket 』
    '\u201D',   # Right double quotation mark "
    '\u2019',   # Right single quotation mark '
    '\u203A',   # Single right-pointing angle quotation mark ›
    '\u00BB',   # Right-pointing double angle quotation mark »
    '\uFF63',   # Halfwidth right corner bracket ｣
}
_ALL_QUOTES = _OPENING_QUOTES | _CLOSING_QUOTES

# Pre-compiled regex patterns
_RE_MULTI_SPACE = re.compile(r'[ \t]+')
_RE_MULTI_NEWLINE = re.compile(r'\n+')
_RE_ALL_WHITESPACE = re.compile(r'\s')



# =============================================================================
# Main Function
# =============================================================================

def normalize(text: str, options: dict | None = None) -> str:
    """
    Normalize Chinese text with specified options.
    
    Args:
        text: Input text to normalize.
        options: Dictionary of normalization options. Only specified options
                 are applied. If None or empty, returns text unchanged.
                 See NormalizeOptions for valid keys and values.
    
    Returns:
        Normalized text.
    
    Raises:
        ImportError: If 'conversion' option is used but OpenCC is not installed.
        ValueError: If an invalid option value is provided.
    
    Examples:
        Basic Unicode normalization:
        
        >>> normalize(text, {'unicode': 'NFC'})
        
        Convert to simplified Chinese:
        
        >>> normalize("軟體開發", {'conversion': 't2s'})
        '软件开发'
        
        Full-width punctuation:
        
        >>> normalize("Hello, world!", {'punctuation': 'full'})
        'Hello，world！'
        
        Smart quotes to corner brackets:
        
        >>> normalize('他说"你好"', {'quotes': 'corner'})
        '他说「你好」'
        
        Combined options:
        
        >>> normalize(text, {
        ...     'conversion': 't2s',
        ...     'unicode': 'NFC',
        ...     'whitespace': 'collapse',
        ...     'punctuation': 'full',
        ...     'quotes': 'smart',
        ... })
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")
    
    if not options:
        return text
    
    result = text
    
    # Order matters for efficiency and correctness:
    
    # 1. Unicode normalization first (affects all subsequent character matching)
    if 'unicode' in options:
        form = options['unicode']
        if form not in ('NFC', 'NFD', 'NFKC', 'NFKD'):
            raise ValueError(f"Invalid unicode form: {form}. "
                           f"Must be one of: NFC, NFD, NFKC, NFKD")
        result = unicodedata.normalize(form, result)
    
    # 2. Script/variant conversion (may change characters significantly)
    if 'conversion' in options:
        result = _convert_script(result, options['conversion'])
    
    # 3. Whitespace normalization
    if 'whitespace' in options:
        mode = options['whitespace']
        if mode not in ('collapse', 'strip', 'remove'):
            raise ValueError(f"Invalid whitespace mode: {mode}. "
                           f"Must be one of: collapse, strip, remove")
        result = _normalize_whitespace(result, mode)
    
    # 4. Width normalization (efficient single-pass with translate)
    if 'numbers' in options:
        mode = options['numbers']
        if mode not in ('full', 'half'):
            raise ValueError(f"Invalid numbers mode: {mode}. "
                           f"Must be one of: full, half")
        if mode == 'half':
            result = result.translate(_TRANS_DIGITS_TO_HALF)
        else:
            result = result.translate(_TRANS_DIGITS_TO_FULL)
    
    if 'letters' in options:
        mode = options['letters']
        if mode not in ('full', 'half'):
            raise ValueError(f"Invalid letters mode: {mode}. "
                           f"Must be one of: full, half")
        if mode == 'half':
            result = result.translate(_TRANS_LETTERS_TO_HALF)
        else:
            result = result.translate(_TRANS_LETTERS_TO_FULL)
    
    # 5. Punctuation width
    if 'punctuation' in options:
        mode = options['punctuation']
        if mode not in ('full', 'half'):
            raise ValueError(f"Invalid punctuation mode: {mode}. "
                           f"Must be one of: full, half")
        if mode == 'full':
            result = result.translate(_TRANS_PUNCT_TO_FULL)
        else:
            result = result.translate(_TRANS_PUNCT_TO_HALF)
    
    # 6. Quote normalization (last, as it's stateful)
    if 'quotes' in options:
        style = options['quotes']
        if style not in _QUOTE_STYLES:
            raise ValueError(f"Invalid quotes style: {style}. "
                           f"Must be one of: {list(_QUOTE_STYLES.keys())}")
        result = _normalize_quotes(result, style)
    
    return result


# =============================================================================
# Internal Functions
# =============================================================================

@lru_cache(maxsize=2)
def _get_opencc_converter(config: str):
    """Get a cached OpenCC converter instance."""
    try:
        import opencc
    except ImportError:
        raise ImportError(
            "Script conversion requires OpenCC. "
            "Install with: pip install opencc"
        )
    return opencc.OpenCC(config)


def _convert_script(text: str, config: str) -> str:
    """Convert between Traditional and Simplified Chinese using OpenCC."""
    converter = _get_opencc_converter(config)
    return converter.convert(text)


def _normalize_whitespace(text: str, mode: str) -> str:
    """Normalize whitespace based on mode."""
    if mode == 'remove':
        return _RE_ALL_WHITESPACE.sub('', text)
    
    if mode == 'strip':
        return '\n'.join(line.strip() for line in text.split('\n'))
    
    if mode == 'collapse':
        # Collapse multiple spaces/tabs → single space, multiple newlines → single newline
        result = _RE_MULTI_SPACE.sub(' ', text)  # spaces and tabs → single space
        result = _RE_MULTI_NEWLINE.sub('\n', result)  # multiple newlines → single newline
        result = '\n'.join(line.strip() for line in result.split('\n'))
        return result.strip()  # strip leading/trailing whitespace from entire text
    
    return text


def _normalize_quotes(text: str, style: str) -> str:
    """
    Normalize quotation marks to specified style with smart nesting.
    
    The first quote type encountered becomes primary, the second type becomes nested.
    Tracks each quote type separately for proper open/close alternation.
    
    For ambiguous quotes (" '), uses count-based alternation (odd = open, even = close).
    """
    open_primary, close_primary, open_nested, close_nested = _QUOTE_STYLES[style]
    
    # Find which quote type appears first
    first_type = None
    for c in text:
        if c in _ALL_QUOTES:
            first_type = _get_quote_type(c)
            break
    
    result = []
    double_depth = 0  # Track double quote nesting
    single_depth = 0  # Track single quote nesting
    
    for i, char in enumerate(text):
        if char in _ALL_QUOTES:
            quote_type = _get_quote_type(char)
            is_opening = _is_opening_quote(text, i)
            
            # Determine if this quote type should use primary or nested style
            use_primary = (quote_type == first_type)
            
            if quote_type == 'double':
                if is_opening is None:
                    is_opening = (double_depth % 2 == 0)
                
                if is_opening:
                    result.append(open_primary if use_primary else open_nested)
                    double_depth += 1
                else:
                    double_depth = max(0, double_depth - 1)
                    result.append(close_primary if use_primary else close_nested)
                    
            else:  # single quote
                if is_opening is None:
                    is_opening = (single_depth % 2 == 0)
                
                if is_opening:
                    result.append(open_primary if use_primary else open_nested)
                    single_depth += 1
                else:
                    single_depth = max(0, single_depth - 1)
                    result.append(close_primary if use_primary else close_nested)
        else:
            result.append(char)
    
    return ''.join(result)


# Double quote characters (used by _get_quote_type)
# Includes: ASCII ", fullwidth ＂, curly " ", corner 「 」, guillemets « »
_DOUBLE_QUOTES = {
    '"',        # ASCII double
    '\uFF02',   # ＂ fullwidth double
    '\u201C',   # " left double
    '\u201D',   # " right double
    '\u300C',   # 「 corner bracket
    '\u300D',   # 」 corner bracket
    '\u00AB',   # « guillemet
    '\u00BB',   # » guillemet
}


def _get_quote_type(char: str) -> str:
    """
    Determine if a quote character is a double or single quote type.
    
    Returns:
        'double' for " ＂ " " 「 」 « »
        'single' for ' ＇ ' ' 『 』 ‹ ›
    """
    if char in _DOUBLE_QUOTES:
        return 'double'
    return 'single'


def _is_opening_quote(text: str, pos: int) -> bool | None:
    """
    Determine if a quote character at position is opening or closing.
    
    Returns:
        True: Definitely opening (e.g., corner bracket 「)
        False: Definitely closing (e.g., corner bracket 」)
        None: Ambiguous (e.g., straight quote ") - let caller decide
    
    For quotes with inherent directionality (「」『』""), we can determine
    direction from the character itself. For ambiguous quotes (" '), we
    return None to signal that depth-based alternation should be used.
    """
    char = text[pos]
    
    # Quotes with inherent directionality - check against known opening/closing sets
    # Note: ASCII " and ' are in both sets (ambiguous)
    only_opening = char in _OPENING_QUOTES and char not in _CLOSING_QUOTES
    only_closing = char in _CLOSING_QUOTES and char not in _OPENING_QUOTES
    
    if only_opening:
        return True
    if only_closing:
        return False
    
    # Ambiguous quotes (" ') - return None to use depth-based alternation
    # This is more reliable than context heuristics which fail in edge cases
    # (e.g., when whitespace is removed and quotes are surrounded by CJK)
    return None


