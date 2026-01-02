---
layout: docs_with_sidebar
title: Text Preprocessing
permalink: /pkg_docs/preprocessing/
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

## Creating a Segmenter

```python
create_segmenter(backend='spacy', strategy='whole', chunk_size=512, 
                 sentence_end_pattern=r"([。！？\.!?……]+)", **kwargs)
```

Factory function to create a segmenter based on the specified backend.

**Parameters:**
- `backend` (str): Segmentation backend (`'spacy'`, `'jieba'`, `'bert'`, or `'llm'`)
- `strategy` (str): Processing strategy
  - `'whole'`: Process entire text at once (default)
  - `'line'`: Split by line breaks and process each line
  - `'sentence'`: Split into sentences and process each sentence
  - `'chunk'`: Split into fixed-size chunks and process each chunk
- `chunk_size` (int): Size of chunks when using `'chunk'` strategy
- `sentence_end_pattern` (str): Regular expression pattern for sentence endings
- `**kwargs`: Additional backend-specific arguments and filters
  - `filters` (dict): Filters to apply during segmentation
    - `'min_word_length'`: Minimum token length (default: 1)
    - `'stopwords'`: List or set of stopwords to exclude
    - `'excluded_pos'`: List or set of POS tags to exclude

**Returns:** An instance of a segmenter (SpacySegmenter, JiebaSegmenter, BertSegmenter, or LLMSegmenter)

## Available Backends

### SpaCy

Uses spaCy NLP library with Chinese models.

**Additional Parameters:**
- `model_name` (str): spaCy model name (`'zh_core_web_sm'`, `'zh_core_web_md'`, or `'zh_core_web_lg'`)
- `disable` (list): Pipeline components to disable for better performance (default: `["ner", "lemmatizer"]`)
- `batch_size` (int): Batch size for processing
- `user_dict` (list or str): Custom user dictionary (list of words or path to dictionary file)

**Installation:** `pip install spacy && python -m spacy download zh_core_web_sm`

### Jieba

Uses the Jieba library for fast segmentation.

**Additional Parameters:**
- `pos_tagging` (bool): Enable POS tagging
- `user_dict_path` (str): Path to custom user dictionary file

**Installation:** `pip install jieba`

### BERT

Uses BERT-based neural segmentation models.

**Additional Parameters:**
- `model_name` (str): BERT model name or path
- `tagging_scheme` (str or list): Tagging scheme (`'be'`, `'bme'`, or `'bmes'`)
- `batch_size` (int): Batch size for processing
- `device` (str): Device to use (`'cpu'` or `'cuda'`)
- `max_sequence_length` (int): Maximum sequence length (default: 512)

**Installation:** `pip install transformers torch`

### LLM

Uses Large Language Models via API services (e.g., OpenAI).

**Additional Parameters:**
- `api_key` (str): API key for the service (required)
- `model` (str): Model name (e.g., `'gpt-3.5-turbo'`)
- `endpoint` (str): API endpoint URL (required)
- `system_message` (str): System message for better segmentation
- `temperature` (float): Temperature for sampling
- `prompt` (str): Custom prompt template

**Installation:** `pip install openai`

## Segmenter Methods

All segmenters have the following method:

```python
segment(text)
```

Segment text into tokens.

**Parameters:**
- `text` (str): Text to segment

**Returns:** 
- If `strategy='whole'`: List of tokens
- If `strategy='line'`, `'sentence'`, or `'chunk'`: List of lists of tokens

## Examples

### Basic Segmentation

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

### With Filters and Custom Dictionary

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

### Integration with Analytics

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
