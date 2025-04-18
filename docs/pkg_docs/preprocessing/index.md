---
layout: default
title: Text Preprocessing
permalink: /pkg_docs/preprocessing/
---

# Text Preprocessing in qhChina

The qhChina package provides preprocessing utilities specifically designed for Chinese text, with a focus on segmentation (word tokenization). The preprocessing module provides flexible tools for tokenizing Chinese text using various backends.

## Text Segmentation

Chinese text segmentation (word tokenization) is a critical preprocessing step for computational text analysis, since Chinese text doesn't use spaces to separate words. The `qhchina.preprocessing` module provides tools for segmenting Chinese text.

### Basic Usage

```python
from qhchina.preprocessing.segmentation import create_segmenter

# Create a segmenter with default settings (uses spaCy)
segmenter = create_segmenter(backend="spacy")

# Segment a single text into tokens
text = "中国经济快速发展"
tokens = segmenter.segment(text)
print(tokens)  # Output: ['中国', '经济', '快速', '发展']

# Segment multiple texts in a batch
texts = ["中国经济快速发展", "人工智能改变世界"]
token_lists = segmenter.segment(texts)
print(token_lists)  # Output: [['中国', '经济', '快速', '发展'], ['人工', '智能', '改变', '世界']]

# Segment text into sentences, each sentence as a list of tokens
long_text = """
中国经济快速发展。人工智能改变世界。
这是中文分词的例子。我们可以把文本分成句子。
"""
sentences = segmenter.segment_to_sentences(long_text)
print(sentences)  # Output: [['中国', '经济', '快速', '发展'], ['人工', '智能', '改变', '世界'], ...]
```

### Creating a Segmenter

The `create_segmenter()` function is the main entry point for creating segmentation tools:

```python
from qhchina.preprocessing.segmentation import create_segmenter

# Create a segmenter with spaCy backend
segmenter = create_segmenter(
    backend="spacy",                    # Segmentation backend
    model_name="zh_core_web_sm",        # Lighter spaCy model
    batch_size=100,                     # Batch size for processing
    filters={
        "min_token_length": 2,          # Minimum token length to include
        "excluded_pos": ["NUM", "SYM"], # POS tags to exclude
        "min_sentence_length": 3        # Minimum sentence length
    }
)

# Or create a segmenter with Jieba backend
jieba_segmenter = create_segmenter(
    backend="jieba",                    # Use Jieba backend
    pos_tagging=True,                   # Enable POS tagging
    filters={
        "min_token_length": 2,          # Minimum token length to include
        "excluded_pos": ["m", "x"],     # POS tags to exclude
        "stopwords": ["的", "了"]       # Stopwords to exclude
    }
)
```

### Available Backends

Currently, the following segmentation backends are supported:

- **spaCy**: A powerful NLP library with Chinese language support
  - Requires installing spaCy and a Chinese model: `pip install spacy && python -m spacy download zh_core_web_sm`
  - Supports POS filtering and other advanced features
  - Slower but more accurate for complex NLP tasks

- **Jieba**: A popular Chinese text segmentation library
  - Requires installing Jieba: `pip install jieba`
  - Faster processing speed, especially for large volumes of text
  - Simpler to use with good accuracy for most use cases

- **BERT**: Neural-based Chinese word segmentation using BERT models
  - Requires installing transformers and torch: `pip install transformers torch`
  - Offers high accuracy for complex texts using deep learning models
  - Supports various tagging schemes and pre-trained models

- **LLM**: Large Language Model-based segmentation using API services like OpenAI
  - Requires installing openai: `pip install openai`
  - Leverages state-of-the-art LLMs for accurate segmentation
  - Customizable through prompts and system messages

### SpacySegmenter

The `SpacySegmenter` class provides Chinese text segmentation using spaCy models:

```python
from qhchina.preprocessing.segmentation import SpacySegmenter

# Create a spaCy-based segmenter
segmenter = SpacySegmenter(
    model_name="zh_core_web_sm",        # spaCy model to use
    disabled=["ner", "lemmatizer"],     # Disable components for speed
    batch_size=100,                     # Batch size for processing
    user_dict=["中国科学院", "人工智能"], # Custom user dictionary
    filters={
        "min_token_length": 2,          # Min token length to keep
        "excluded_pos": ["NUM", "SYM", "SPACE"],  # POS tags to exclude
        "min_sentence_length": 3        # Min sentence length to keep
    }
)
```

#### Available spaCy Models for Chinese

| Model | Size | Description |
|-------|------|-------------|
| `zh_core_web_sm` | Small | Basic POS tagging and dependency parsing |
| `zh_core_web_md` | Medium | Includes word vectors |
| `zh_core_web_lg` | Large | Larger vocabulary and word vectors |

Install with: `python -m spacy download zh_core_web_sm`

### JiebaSegmenter

The `JiebaSegmenter` class provides Chinese text segmentation using the Jieba library:

```python
from qhchina.preprocessing.segmentation import JiebaSegmenter

# Create a Jieba-based segmenter
segmenter = JiebaSegmenter(
    pos_tagging=True,                   # Enable POS tagging
    user_dict_path="path/to/dict.txt",  # Custom user dictionary
    filters={
        "min_token_length": 2,          # Min token length to keep
        "excluded_pos": ["m", "x"],     # POS tags to exclude (Jieba's POS tags)
        "stopwords": ["的", "了"],      # Words to exclude
        "min_sentence_length": 3        # Min sentence length to keep
    }
)
```

### BertSegmenter

The `BertSegmenter` class provides neural-based Chinese text segmentation using BERT models:

```python
from qhchina.preprocessing.segmentation import BertSegmenter

# Create a BERT-based segmenter with a fine-tuned model
segmenter = BertSegmenter(
    model_name="bert-modern-chinese-segmentation",   # BERT model to use
    tagging_scheme="bmes",              # Tagging scheme: "be", "bme", or "bmes"
    batch_size=16,                      # Batch size for processing
    device="cuda",                      # Use GPU if available
    filters={
        "min_token_length": 2,          # Min token length to keep
        "stopwords": ["的", "了"],      # Words to exclude
        "min_sentence_length": 3        # Min sentence length to keep
    }
)

# Segment text
text = "中国经济快速发展，人工智能改变世界。"
tokens = segmenter.segment(text)
print(tokens)  # Output: ['中国', '经济', '快速', '发展', '，', '人工智能', '改变', '世界', '。']

# Segment to sentences
sentences = segmenter.segment_to_sentences(text)
print(sentences)  # Output: [['中国', '经济', '快速', '发展', '，', '人工智能', '改变', '世界', '。']]
```

#### Tagging Schemes

The BertSegmenter supports several tagging schemes:

| Scheme | Tags | Description |
|--------|------|-------------|
| `be` | B, E | Beginning and End of words |
| `bme` | B, M, E | Beginning, Middle, and End of words |
| `bmes` | B, M, E, S | Beginning, Middle, End, and Single-character words |

### LLMSegmenter

The `LLMSegmenter` class provides Chinese text segmentation using Large Language Models via API services:

```python
from qhchina.preprocessing.segmentation import LLMSegmenter

# Create an LLM-based segmenter using OpenAI
segmenter = LLMSegmenter(
    api_key="your-openai-api-key",      # API key for the service
    model="deepseek-chat",              # Model to use
    endpoint="https://api.deepseek.com", 
    system_message="你是语言学专家。",
    temperature=0.1,                    # Lower temperature for more consistent results
    output_dir="./segmentation_results", # Save results to this directory
    filters={
        "min_token_length": 1,          # Min token length to keep
        "stopwords": ["的", "了"]       # Words to exclude
    }
)

# Segment text
text = "人工智能大模型能够准确分词。"
tokens = segmenter.segment(text)
print(tokens)  # Output: ['人工智能', '大', '模型', '能够', '准确', '分词', '。']

# Process multiple texts
texts = ["中国经济快速发展", "人工智能改变世界"]
token_lists = segmenter.segment(texts)
print(token_lists)

# Segment to sentences
long_text = "今天天气真好，我们去散步吧！这是中文分词的例子。"
sentences = segmenter.segment_to_sentences(long_text)
print(sentences)  # Output: [['今天', '天气', '真', '好', '，', '我们', '去', '散步', '吧', '！'], ['这是', '中文', '分词', '的', '例子', '。']]
```

#### Custom Prompts

You can customize the segmentation prompt for different segmentation styles:

```python
# Custom prompt for academic-style segmentation
custom_prompt = """
请将以下中文文本分词，按照学术标准分词。请用JSON格式回答。

示例:
输入: "人工智能大模型能够准确分词"
输出: ["人工智能", "大模型", "能够", "准确", "分词"]

输入: "{text}"
输出:
"""

segmenter = LLMSegmenter(
    api_key="your-openai-api-key",
    model="gpt-4",
    prompt=custom_prompt,
    temperature=0
)
```

## Filtering Options

All segmenters support filtering options that can be passed during initialization:

| Filter | Description |
|--------|-------------|
| `min_token_length` | Minimum length of tokens to include (default: 1) |
| `excluded_pos` | Set of POS tags to exclude (requires POS tagging support) |
| `min_sentence_length` | Minimum sentence length to include when using `segment_to_sentences()` |
| `stopwords` | List of words to exclude from results |

## Common Workflows

### Basic Text Processing Pipeline

```python
from qhchina.preprocessing.segmentation import create_segmenter
from qhchina.helpers import load_texts, load_stopwords

# Load stopwords
stopwords = load_stopwords("zh_sim")

# Create segmenter with filters
segmenter = create_segmenter(
    backend="jieba",  # Using Jieba for faster processing
    filters={"stopwords": stopwords, "min_token_length": 2}
)

# Load and process texts
raw_texts = load_texts(["path/to/file1.txt", "path/to/file2.txt"])
processed_documents = []

for text in raw_texts:
    # Segment text into sentences
    sentences = segmenter.segment_to_sentences(text)
    processed_documents.extend(sentences)
    
# Now processed_documents is ready for analytics tasks
```

### Integration with Analytics

```python
from qhchina.preprocessing.segmentation import create_segmenter
from qhchina.analytics.topicmodels import LDAGibbsSampler

# Create segmenter
segmenter = create_segmenter(backend="spacy")

# Process text
text = """
中国经济快速发展。改革开放带来巨大变化。
科技创新促进经济增长。人工智能改变世界。
"""
sentences = segmenter.segment_to_sentences(text)

# Use processed text in analytics
lda = LDAGibbsSampler(n_topics=2)
lda.fit(sentences)

# Get topics
topics = lda.get_topics(n_words=5)
for i, topic in enumerate(topics):
    print(f"Topic {i}: {[word for word, _ in topic]}")
```