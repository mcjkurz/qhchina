---
layout: docs_with_sidebar
title: Text Preprocessing
permalink: /pkg_docs/preprocessing/
functions:
  - name: SegmentationWrapper
    anchor: segmentationwrapper
  - name: SegmentationWrapper.close()
    anchor: segmentationwrapper-close
  - name: SegmentationWrapper.reset_user_dict()
    anchor: segmentationwrapper-reset_user_dict
  - name: SegmentationWrapper.segment()
    anchor: segmentationwrapper-segment
  - name: SpacySegmenter
    anchor: spacysegmenter
  - name: SpacySegmenter.reset_user_dict()
    anchor: spacysegmenter-reset_user_dict
  - name: PKUSegmenter
    anchor: pkusegmenter
  - name: PKUSegmenter.reset_user_dict()
    anchor: pkusegmenter-reset_user_dict
  - name: JiebaSegmenter
    anchor: jiebasegmenter
  - name: JiebaSegmenter.reset_user_dict()
    anchor: jiebasegmenter-reset_user_dict
  - name: BertSegmenter
    anchor: bertsegmenter
  - name: LLMSegmenter
    anchor: llmsegmenter
  - name: create_segmenter()
    anchor: create_segmenter
has_examples: True
import_from: qhchina.preprocessing.segmentation
---

# Text Preprocessing

The `qhchina.preprocessing` module provides Chinese text segmentation (word tokenization) with various backends and processing strategies.

```python
from qhchina.preprocessing.segmentation import create_segmenter

segmenter = create_segmenter(backend="spacy", strategy="sentence")
sentences = segmenter.segment("深度学习正在改变世界。自然语言处理是其中一个领域。")
# [['深度', '学习', '正在', '改变', '世界', '。'], ['自然', '语言', '处理', '是', '其中', '一个', '领域', '。']]
```

---

## API Reference

<!-- API-START -->

<h3 id="segmentationwrapper">SegmentationWrapper</h3>

```python
SegmentationWrapper(
    strategy: str = 'whole',
    chunk_size: int = 512,
    filters: Dict[str, Any] = None,
    user_dict: Union[str, List[Union[str, Tuple]], NoneType] = None,
    sentence_end_pattern: str = '([。！？\\.!?……]+)'
)
```

Base segmentation wrapper class that can be extended for different segmentation tools.

**Parameters:**
- `strategy`: Strategy to process texts. Options: 'line', 'sentence', 'chunk', 'whole'. 
  Default is 'whole'.
- `chunk_size`: Size of chunks when using 'chunk' strategy.
- `filters`: Dictionary of filters to apply during segmentation:
  - stopwords: List or set of stopwords to exclude (converted to set internally)
  - min_word_length: Minimum length of tokens to include (default 1)
  - excluded_pos: List or set of POS tags to exclude (converted to set internally)
- `user_dict`: Custom user dictionary for segmentation. Can be:
  - str: Path to a dictionary file
  - List[str]: List of words
  - List[Tuple]: List of tuples like (word, freq, pos) or (word, freq)
- `sentence_end_pattern`: Regular expression pattern for sentence endings (default: 
  Chinese and English punctuation).

<h4 id="segmentationwrapper-close">SegmentationWrapper.close()</h4>

```python
close()
```

Clean up resources. Call this when done with the segmenter.

<h4 id="segmentationwrapper-reset_user_dict">SegmentationWrapper.reset_user_dict()</h4>

```python
reset_user_dict()
```

Reset the user dictionary to default state.

This clears any custom words that were added via user_dict.
Subclasses should override this method to implement backend-specific reset logic.

<h4 id="segmentationwrapper-segment">SegmentationWrapper.segment()</h4>

```python
segment(text: str)
```

Segment text into tokens based on the selected strategy.

**Parameters:**
- `text`: Text to segment

**Returns:**
If strategy is 'whole': A single list of tokens
If strategy is 'line', 'sentence', or 'chunk': A list of lists, where each inner list
contains tokens for a line, sentence, or chunk respectively

<br>

<h3 id="spacysegmenter">SpacySegmenter</h3>

```python
SpacySegmenter(
    model_name: str = 'zh_core_web_sm',
    disable: Optional[List[str]] = None,
    batch_size: int = 200,
    user_dict: Union[str, List[Union[str, Tuple]], NoneType] = None,
    strategy: str = 'whole',
    chunk_size: int = 512,
    filters: Dict[str, Any] = None,
    sentence_end_pattern: str = '([。！？\\.!?……]+)'
)
```

Segmentation wrapper for spaCy models.

Note: spaCy Chinese models use spacy-pkuseg, a fork of pkuseg trained on the OntoNotes
corpus and co-trained with downstream statistical components (POS tagging, NER, parsing).

**Parameters:**
- `model_name`: Name of the spaCy model to use.
- `disable`: List of pipeline components to disable for better performance; 
  For common applications, use ["ner", "lemmatizer"]. Default is None.
- `batch_size`: Batch size for processing multiple texts.
- `user_dict`: Custom user dictionary - either a list of words/tuples or path to a 
  dictionary file.
- `strategy`: Strategy to process texts. Options: 'line', 'sentence', 'chunk', 'whole'.
- `chunk_size`: Size of chunks when using 'chunk' strategy.
- `filters`: Dictionary of filters to apply during segmentation:
  - min_word_length: Minimum length of tokens to include (default 1)
  - excluded_pos: Set of POS tags to exclude from token outputs
  - stopwords: Set of stopwords to exclude
- `sentence_end_pattern`: Regular expression pattern for sentence endings.

<h4 id="spacysegmenter-reset_user_dict">SpacySegmenter.reset_user_dict()</h4>

```python
reset_user_dict()
```

Reset the spaCy tokenizer's user dictionary.

This clears any custom words that were added via pkuseg_update_user_dict.
Note: This resets to an empty user dictionary, not the original state if one was loaded.

<br>

<h3 id="pkusegmenter">PKUSegmenter</h3>

```python
PKUSegmenter(
    model_name: str = 'default',
    user_dict: Union[str, List[Union[str, Tuple]], NoneType] = None,
    pos_tagging: bool = False,
    strategy: str = 'whole',
    chunk_size: int = 512,
    filters: Dict[str, Any] = None,
    sentence_end_pattern: str = '([。！？\\.!?……]+)'
)
```

Segmentation wrapper for PKUSeg Chinese text segmentation.

PKUSeg is a toolkit for multi-domain Chinese word segmentation developed by
Peking University. It uses the original pkuseg package with its own pre-trained
models (different from spacy-pkuseg, which is trained on OntoNotes).

Note: PKUSeg does not support dynamic user dictionary updates. The user dictionary
is loaded at initialization time. To change the dictionary, call reset_user_dict()
which will reinitialize the segmenter.

**Parameters:**
- `model_name`: Name of the model to use. Options:
  - 'default': General domain model (default)
  - 'news': News domain
  - 'web': Web domain  
  - 'medicine': Medical domain
  - 'tourism': Tourism domain
  - Or a path to a custom model directory
- `user_dict`: Custom user dictionary. Can be:
  - str: Path to a dictionary file (one word per line)
  - List[str]: List of words
  - List[Tuple]: List of tuples (only first element/word is used)
- `pos_tagging`: Whether to include POS tagging in segmentation.
- `strategy`: Strategy to process texts. Options: 'line', 'sentence', 'chunk', 'whole'.
- `chunk_size`: Size of chunks when using 'chunk' strategy.
- `filters`: Dictionary of filters to apply during segmentation:
  - min_word_length: Minimum length of tokens to include (default 1)
  - excluded_pos: List of POS tags to exclude (if pos_tagging is True)
  - stopwords: Set of stopwords to exclude
- `sentence_end_pattern`: Regular expression pattern for sentence endings.

<h4 id="pkusegmenter-reset_user_dict">PKUSegmenter.reset_user_dict()</h4>

```python
reset_user_dict()
```

Reset the user dictionary by reinitializing PKUSeg without a user dict.

Note: PKUSeg doesn't support dynamic dictionary updates, so we reinitialize
the entire segmenter. This is different from Jieba where we can reset the
global state.

<br>

<h3 id="jiebasegmenter">JiebaSegmenter</h3>

```python
JiebaSegmenter(
    user_dict: Union[str, List[Union[str, Tuple]], NoneType] = None,
    pos_tagging: bool = False,
    strategy: str = 'whole',
    chunk_size: int = 512,
    filters: Dict[str, Any] = None,
    sentence_end_pattern: str = '([。！？\\.!?……]+)'
)
```

Segmentation wrapper for Jieba Chinese text segmentation.

**Parameters:**
- `user_dict`: Custom user dictionary for Jieba. Can be:
  - str: Path to a dictionary file
  - List[str]: List of words
  - List[Tuple]: List of tuples like (word, freq, pos) or (word, freq)
- `pos_tagging`: Whether to include POS tagging in segmentation.
- `strategy`: Strategy to process texts. Options: 'line', 'sentence', 'chunk', 'whole'.
- `chunk_size`: Size of chunks when using 'chunk' strategy.
- `filters`: Dictionary of filters to apply during segmentation:
  - min_word_length: Minimum length of tokens to include (default 1)
  - excluded_pos: List of POS tags to exclude (if pos_tagging is True)
  - stopwords: Set of stopwords to exclude
- `sentence_end_pattern`: Regular expression pattern for sentence endings.

<h4 id="jiebasegmenter-reset_user_dict">JiebaSegmenter.reset_user_dict()</h4>

```python
reset_user_dict()
```

Reset Jieba's dictionary to default state.

This reinitializes Jieba, clearing any custom words that were added.
Note: Jieba uses a global state, so this affects all JiebaSegmenter instances.

<br>

<h3 id="bertsegmenter">BertSegmenter</h3>

```python
BertSegmenter(
    model_name: str = None,
    model=None,
    tokenizer=None,
    tagging_scheme: Union[str, List[str]] = 'be',
    batch_size: int = 32,
    device: Optional[str] = None,
    remove_special_tokens: bool = True,
    max_sequence_length: int = 512,
    user_dict: Union[str, List[Union[str, Tuple]], NoneType] = None,
    strategy: str = 'whole',
    chunk_size: int = 512,
    filters: Dict[str, Any] = None,
    sentence_end_pattern: str = '([。！？\\.!?……]+)'
)
```

Segmentation wrapper for BERT-based Chinese word segmentation.

**Parameters:**
- `model_name`: Name of the pre-trained BERT model to load (optional if model and 
  tokenizer are provided).
- `model`: Pre-initialized model instance (optional if model_name is provided).
- `tokenizer`: Pre-initialized tokenizer instance (optional if model_name is provided).
- `tagging_scheme`: Either a string ('be', 'bmes') or a list of tags in their exact 
  order (e.g. ["B", "E"]). When a list is provided, the order of tags matters 
  as it maps to prediction indices.
- `batch_size`: Batch size for processing.
- `device`: Device to use ('cpu', 'cuda', etc.).
- `remove_special_tokens`: Whether to remove special tokens (CLS, SEP) from output. 
  Default is True, which works for BERT-based models.
- `max_sequence_length`: Maximum sequence length for BERT models (default 512). If 
  the text is longer than this, it will be split into chunks.
- `user_dict`: Custom user dictionary (not supported for BERT segmenter, will be ignored
  with a warning).
- `strategy`: Strategy to process texts. Options: 'line', 'sentence', 'chunk', 'whole'.
- `chunk_size`: Size of chunks when using 'chunk' strategy.
- `filters`: Dictionary of filters to apply during segmentation:
  - min_word_length: Minimum length of tokens to include (default 1)
  - excluded_pos: Set of POS tags to exclude from token outputs
  - stopwords: Set of stopwords to exclude
- `sentence_end_pattern`: Regular expression pattern for sentence endings.

<br>

<h3 id="llmsegmenter">LLMSegmenter</h3>

```python
LLMSegmenter(
    api_key: str,
    model: str,
    endpoint: str,
    prompt: str = None,
    system_message: str = None,
    temperature: float = 1,
    max_tokens: int = 2048,
    retry_patience: int = 1,
    timeout: float = 60.0,
    user_dict: Union[str, List[Union[str, Tuple]], NoneType] = None,
    strategy: str = 'whole',
    chunk_size: int = 512,
    filters: Dict[str, Any] = None,
    sentence_end_pattern: str = '([。！？\\.!?……]+)'
)
```

Segmentation wrapper using Language Model APIs like OpenAI.

**Parameters:**
- `api_key`: API key for the language model service.
- `model`: Model name to use.
- `endpoint`: API endpoint URL.
- `prompt`: Custom prompt template with {text} placeholder (if None, uses DEFAULT_PROMPT).
- `system_message`: Optional system message to prepend to API calls.
- `temperature`: Temperature for model sampling (lower for more deterministic output).
- `max_tokens`: Maximum tokens in the response.
- `retry_patience`: Number of retries for API calls (default 1, meaning 1 retry = 
  2 total attempts).
- `timeout`: Timeout in seconds for API calls (default 60.0). Set to None for no timeout.
- `user_dict`: Custom user dictionary (not supported for LLM segmenter, will be ignored
  with a warning).
- `strategy`: Strategy to process texts. Options: 'line', 'sentence', 'chunk', 'whole'.
- `chunk_size`: Size of chunks when using 'chunk' strategy.
- `filters`: Dictionary of filters to apply during segmentation:
  - min_word_length: Minimum length of tokens to include (default 1)
  - excluded_pos: Set of POS tags to exclude from token outputs
  - stopwords: Set of stopwords to exclude
- `sentence_end_pattern`: Regular expression pattern for sentence endings.

<br>

<h3 id="create_segmenter">create_segmenter()</h3>

```python
create_segmenter(
    backend: str = 'spacy',
    strategy: str = 'whole',
    chunk_size: int = 512,
    sentence_end_pattern: str = '([。！？\\.!?……]+)',
    **kwargs
)
```

Create a segmenter based on the specified backend.

**Parameters:**
- `backend`: The segmentation backend to use ('spacy', 'pkuseg', 'jieba', 'bert', 'llm')
- `strategy`: Strategy to process texts ['line', 'sentence', 'chunk', 'whole']
- `chunk_size`: Size of chunks when using 'chunk' strategy
- `sentence_end_pattern`: Regular expression pattern for sentence endings (default: Chinese and English punctuation)
- `**kwargs`: Additional arguments to pass to the segmenter constructor
  - user_dict: Custom user dictionary. Can be:
      - str: Path to a dictionary file
      - List[str]: List of words
      - List[Tuple]: List of tuples like (word, freq, pos) or (word, freq)
      Note: Not supported for 'bert' and 'llm' backends (will log a warning)
  - filters: Dictionary of filters to apply during segmentation
      - min_word_length: Minimum length of tokens to include (default 1)
      - stopwords: Set of stopwords to exclude
      - excluded_pos: Set of POS tags to exclude (for backends that support POS tagging)
  - retry_patience: (LLM backend only) Number of retry attempts for API calls (default 1)
  - timeout: (LLM backend only) Timeout in seconds for API calls (default 60.0)
  - Other backend-specific arguments

**Returns:**
An instance of a SegmentationWrapper subclass

**Raises:**
- `ValueError`: If the specified backend is not supported

<br>

<!-- API-END -->

---

## Examples

**Basic Segmentation**

```python
from qhchina.preprocessing.segmentation import create_segmenter

# Create a segmenter with default settings
segmenter = create_segmenter(backend="spacy")

# Segment a single text
text = "量子计算将改变密码学的未来"
tokens = segmenter.segment(text)
print(tokens)
# Output: ['量子', '计算', '将', '改变', '密码学', '的', '未来']

# Process sentence by sentence
segmenter = create_segmenter(backend="spacy", strategy="sentence")
long_text = """古代文明的天文观测记录。
量子纠缠现象的神奇特性。
人类意识的哲学讨论。"""
sentences = segmenter.segment(long_text)
# Output: [['古代', '文明', '的', '天文', '观测', '记录', '。'], ...]
```

**With Filters and Custom Dictionary**

```python
from qhchina.helpers import load_stopwords

# Load stopwords
stopwords = load_stopwords("zh_sim")

# Create segmenter with filters
segmenter = create_segmenter(
    backend="spacy",
    model_name="zh_core_web_sm",
    strategy="sentence",
    user_dict=["量子计算", "深度学习"],
    filters={
        "min_word_length": 2,
        "excluded_pos": ["NUM", "SYM"],
        "stopwords": stopwords
    }
)

# Segment text
text = "深度学习模型理解复杂语境。量子计算改变加密技术。"
sentences = segmenter.segment(text)
```

**Integration with Analytics**

```python
from qhchina.preprocessing.segmentation import create_segmenter
from qhchina.analytics.topicmodels import LDAGibbsSampler
from qhchina.helpers import load_texts, load_stopwords

# Load stopwords
stopwords = load_stopwords("zh_sim")

# Create segmenter
segmenter = create_segmenter(
    backend="jieba",
    strategy="sentence",
    filters={"stopwords": stopwords, "min_word_length": 2}
)

# Load and process texts
texts = load_texts(["file1.txt", "file2.txt"])
all_sentences = []
for text in texts:
    sentences = segmenter.segment(text)
    all_sentences.extend(sentences)

# Use with topic modeling
lda = LDAGibbsSampler(n_topics=10, min_word_count=5)
lda.fit(all_sentences)

# Get topics
topics = lda.get_topics(n_words=10)
```
