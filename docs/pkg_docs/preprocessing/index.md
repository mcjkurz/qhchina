---
layout: docs_with_sidebar
title: Text Preprocessing
permalink: /pkg_docs/preprocessing/
functions:
  - name: SegmentationWrapper
    anchor: segmentationwrapper
  - name: SegmentationWrapper.segment()
    anchor: segmentationwrapper-segment
  - name: SpacySegmenter
    anchor: spacysegmenter
  - name: JiebaSegmenter
    anchor: jiebasegmenter
  - name: BertSegmenter
    anchor: bertsegmenter
  - name: LLMSegmenter
    anchor: llmsegmenter
  - name: create_segmenter()
    anchor: create_segmenter
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
    sentence_end_pattern: str = '([。！？\\.!?……]+)'
)
```

Base segmentation wrapper class that can be extended for different segmentation tools.

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
    model_name: str = 'zh_core_web_lg',
    disable: Optional[List[str]] = None,
    batch_size: int = 200,
    user_dict: Union[List[str], str] = None,
    strategy: str = 'whole',
    chunk_size: int = 512,
    filters: Dict[str, Any] = None,
    sentence_end_pattern: str = '([。！？\\.!?……]+)'
)
```

Segmentation wrapper for spaCy Chinese models.

Uses spaCy NLP library with Chinese language models for tokenization.
Supports custom user dictionaries and POS-based filtering.

**Parameters:**
- `model_name` (str): spaCy model name. Options include:
  - 'zh_core_web_sm': Small model (~50MB), fastest but less accurate
  - 'zh_core_web_md': Medium model (~90MB), balanced
  - 'zh_core_web_lg': Large model (~560MB), most accurate (default)
- `disable` (list): Pipeline components to disable for better performance.
  Default: ["ner", "lemmatizer"]. Set to [] to enable all components.
- `batch_size` (int): Batch size for processing multiple texts (default: 200)
- `user_dict` (list or str): Custom user dictionary for domain-specific terms.
  Can be a list of words or path to a dictionary file (one word per line).
- `strategy` (str): Text processing strategy - 'whole', 'line', 'sentence', or 'chunk'
- `chunk_size` (int): Size of chunks when using 'chunk' strategy (default: 512)
- `filters` (dict): Filters to apply during segmentation:
  - min_word_length: Minimum token length (default: 1)
  - stopwords: Set of stopwords to exclude
  - excluded_pos: Set of POS tags to exclude (e.g., {'PUNCT', 'SPACE'})
- `sentence_end_pattern` (str): Regex pattern for sentence endings

**Example:**
```python
>>> from qhchina.preprocessing import create_segmenter
>>> segmenter = create_segmenter(
...     backend="spacy",
...     model_name="zh_core_web_sm",
...     user_dict=["深度学习", "自然语言处理"]
... )
>>> tokens = segmenter.segment("深度学习正在改变世界")
>>> print(tokens)
['深度学习', '正在', '改变', '世界']
```

<br>

<h3 id="jiebasegmenter">JiebaSegmenter</h3>

```python
JiebaSegmenter(
    user_dict_path: str = None,
    pos_tagging: bool = False,
    strategy: str = 'whole',
    chunk_size: int = 512,
    filters: Dict[str, Any] = None,
    sentence_end_pattern: str = '([。！？\\.!?……]+)'
)
```

Segmentation wrapper for Jieba Chinese text segmentation.

Uses the Jieba library for fast, dictionary-based Chinese word segmentation.
Jieba is lightweight and fast, making it suitable for large-scale processing.

**Parameters:**
- `user_dict_path` (str): Path to a custom user dictionary file.
  The file should contain one word per line, optionally with frequency and POS tag:
  "深度学习 5 n" or just "深度学习"
- `pos_tagging` (bool): Enable POS tagging during segmentation (default: False).
  When enabled, allows filtering by POS tags using the 'excluded_pos' filter.
- `strategy` (str): Text processing strategy - 'whole', 'line', 'sentence', or 'chunk'
- `chunk_size` (int): Size of chunks when using 'chunk' strategy (default: 512)
- `filters` (dict): Filters to apply during segmentation:
  - min_word_length: Minimum token length (default: 1)
  - stopwords: Set of stopwords to exclude
  - excluded_pos: Set of POS tags to exclude (requires pos_tagging=True)
- `sentence_end_pattern` (str): Regex pattern for sentence endings

**Example:**
```python
>>> from qhchina.preprocessing import create_segmenter
>>> segmenter = create_segmenter(backend="jieba", pos_tagging=True)
>>> tokens = segmenter.segment("我爱北京天安门")
>>> print(tokens)
['我', '爱', '北京', '天安门']

>>> # With POS filtering
>>> segmenter = create_segmenter(
...     backend="jieba",
...     pos_tagging=True,
...     filters={"excluded_pos": {"x", "w"}}  # Exclude punctuation
... )
```

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
    strategy: str = 'whole',
    chunk_size: int = 512,
    filters: Dict[str, Any] = None,
    sentence_end_pattern: str = '([。！？\\.!?……]+)'
)
```

Segmentation wrapper for BERT-based Chinese word segmentation.

Uses BERT or other transformer models fine-tuned for Chinese word segmentation
as a sequence labeling task. Supports various tagging schemes (BE, BME, BMES).

**Parameters:**
- `model_name` (str): HuggingFace model name or local path to a fine-tuned model.
  Must be a model trained for token classification with the specified tagging scheme.
- `model`: Pre-initialized model instance (alternative to model_name)
- `tokenizer`: Pre-initialized tokenizer instance (alternative to model_name)
- `tagging_scheme` (str or list): Tagging scheme for word boundary prediction:
  - 'be': 2-tag scheme (Beginning, End)
  - 'bme': 3-tag scheme (Beginning, Middle, End)
  - 'bmes': 4-tag scheme (Beginning, Middle, End, Single)
  - Or provide a custom list like ["B", "I", "O"]
- `batch_size` (int): Batch size for processing (default: 32)
- `device` (str): Device to use - 'cpu', 'cuda', or 'cuda:0' (auto-detected if None)
- `remove_special_tokens` (bool): Remove [CLS] and [SEP] from output (default: True)
- `max_sequence_length` (int): Maximum sequence length for BERT (default: 512)
- `strategy` (str): Text processing strategy - 'whole', 'line', 'sentence', or 'chunk'
- `chunk_size` (int): Size of chunks when using 'chunk' strategy (default: 512)
- `filters` (dict): Filters to apply during segmentation:
  - min_word_length: Minimum token length (default: 1)
  - stopwords: Set of stopwords to exclude
- `sentence_end_pattern` (str): Regex pattern for sentence endings

**Example:**
```python
>>> from qhchina.preprocessing import create_segmenter
>>> # Using a pre-trained Chinese word segmentation model
>>> segmenter = create_segmenter(
...     backend="bert",
...     model_name="bert-base-chinese-cws",
...     tagging_scheme="bmes",
...     device="cuda"
... )
>>> tokens = segmenter.segment("自然语言处理是人工智能的重要领域")
```

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
    strategy: str = 'whole',
    chunk_size: int = 512,
    filters: Dict[str, Any] = None,
    sentence_end_pattern: str = '([。！？\\.!?……]+)'
)
```

Segmentation wrapper using Large Language Model APIs.

Uses LLM APIs (OpenAI, Azure OpenAI, or compatible endpoints) for Chinese word
segmentation via prompting. Useful when high-quality segmentation is needed
and API costs are acceptable.

**Parameters:**
- `api_key` (str): API key for the LLM service (required)
- `model` (str): Model name to use, e.g., 'gpt-3.5-turbo', 'gpt-4' (required)
- `endpoint` (str): API endpoint URL (required). Examples:
  - OpenAI: "https://api.openai.com/v1"
  - Azure: "https://your-resource.openai.azure.com/openai/deployments/your-deployment"
- `prompt` (str): Custom prompt template with {text} placeholder.
  If None, uses a default Chinese segmentation prompt.
- `system_message` (str): Optional system message for the API call
- `temperature` (float): Sampling temperature (default: 1). Lower values = more deterministic.
- `max_tokens` (int): Maximum tokens in the response (default: 2048)
- `retry_patience` (int): Number of retry attempts on API failure (default: 1)
- `timeout` (float): Timeout in seconds for API calls (default: 60.0)
- `strategy` (str): Text processing strategy - 'whole', 'line', 'sentence', or 'chunk'
- `chunk_size` (int): Size of chunks when using 'chunk' strategy (default: 512)
- `filters` (dict): Filters to apply during segmentation:
  - min_word_length: Minimum token length (default: 1)
  - stopwords: Set of stopwords to exclude
- `sentence_end_pattern` (str): Regex pattern for sentence endings

**Example:**
```python
>>> from qhchina.preprocessing import create_segmenter
>>> import os
>>> segmenter = create_segmenter(
...     backend="llm",
...     api_key=os.environ["OPENAI_API_KEY"],
...     model="gpt-3.5-turbo",
...     endpoint="https://api.openai.com/v1",
...     temperature=0.1  # Lower for more consistent output
... )
>>> tokens = segmenter.segment("量子计算将改变世界")
```

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

Factory function to create a segmenter based on the specified backend.

This is the recommended way to create segmenters, as it provides a unified
interface for all supported backends.

**Parameters:**
- `backend` (str): The segmentation backend to use:
  - 'spacy': spaCy with Chinese models (recommended for accuracy)
  - 'jieba': Jieba (fast, dictionary-based)
  - 'bert': BERT-based neural segmentation
  - 'llm': Large Language Model API-based segmentation
- `strategy` (str): How to process the input text:
  - 'whole': Process entire text at once, return flat list of tokens
  - 'line': Split by newlines, return list of token lists
  - 'sentence': Split by sentence boundaries, return list of token lists
  - 'chunk': Split into fixed-size chunks, return list of token lists
- `chunk_size` (int): Size of chunks when using 'chunk' strategy (default: 512)
- `sentence_end_pattern` (str): Regex for sentence boundaries (default: Chinese/English punctuation)
- `**kwargs`: Backend-specific arguments:
  Common filters (all backends):
      - filters (dict): Filtering options
          - min_word_length: Minimum token length (default: 1)
          - stopwords: Set/list of stopwords to exclude
          - excluded_pos: Set of POS tags to exclude
  
  SpaCy backend:
      - model_name: 'zh_core_web_sm', 'zh_core_web_md', or 'zh_core_web_lg'
      - disable: Pipeline components to disable (default: ["ner", "lemmatizer"])
      - batch_size: Processing batch size (default: 200)
      - user_dict: Custom dictionary (list of words or file path)
  
  Jieba backend:
      - user_dict_path: Path to custom dictionary file
      - pos_tagging: Enable POS tagging (default: False)
  
  BERT backend:
      - model_name: HuggingFace model name or path
      - tagging_scheme: 'be', 'bme', 'bmes', or custom list
      - batch_size: Processing batch size (default: 32)
      - device: 'cpu' or 'cuda' (auto-detected if None)
  
  LLM backend:
      - api_key: API key (required)
      - model: Model name like 'gpt-3.5-turbo' (required)
      - endpoint: API endpoint URL (required)
      - temperature: Sampling temperature (default: 1)
      - retry_patience: Retry attempts (default: 1)
      - timeout: API timeout in seconds (default: 60.0)

**Returns:**
(SegmentationWrapper) An instance of SpacySegmenter, JiebaSegmenter,
BertSegmenter, or LLMSegmenter based on the backend.

**Raises:**
- `ValueError`: If the specified backend is not supported

**Example:**
```python
>>> from qhchina.preprocessing import create_segmenter
>>> from qhchina.helpers import load_stopwords
>>> 
>>> # Basic usage with spaCy
>>> segmenter = create_segmenter(backend="spacy")
>>> tokens = segmenter.segment("深度学习正在改变世界")
>>> 
>>> # With sentence splitting and stopword filtering
>>> stopwords = load_stopwords("zh_sim")
>>> segmenter = create_segmenter(
...     backend="jieba",
...     strategy="sentence",
...     filters={"stopwords": stopwords, "min_word_length": 2}
... )
>>> sentences = segmenter.segment("第一句话。第二句话。")
```

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
