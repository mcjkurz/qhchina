---
layout: docs_with_sidebar
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
text = "量子计算将改变密码学的未来"
tokens = segmenter.segment(text)
print(tokens)  # Output: ['量子', '计算', '将', '改变', '密码学', '的', '未来']

# To process multiple texts, iterate over them
texts = ["深度学习模型理解复杂语境", "太空探索发现新的系外行星"]
results = []
for text in texts:
    tokens = segmenter.segment(text)
    results.append(tokens)
print(results)  # Output: [['深度', '学习', '模型', '理解', '复杂', '语境'], ['太空', '探索', '发现', '新的', '系外', '行星']]

# Create a segmenter with line-by-line processing strategy
line_segmenter = create_segmenter(backend="spacy", strategy="line")
long_text = """古代文明的天文观测记录。
量子纠缠现象的神奇特性。
人类意识的哲学讨论。"""
result = line_segmenter.segment(long_text)  # Process each line separately
print(result)  # Output: [['古代', '文明', '的', '天文', '观测', '记录', '。'], ['量子', '纠缠', '现象', '的', '神奇', '特性', '。'], ...]
```

### Segmentation Strategies

The segmenters support different processing strategies through the `strategy` parameter:

```python
from qhchina.preprocessing.segmentation import create_segmenter

# Create a segmenter with sentence-by-sentence processing
segmenter = create_segmenter(
    backend="spacy",
    strategy="sentence"  # Process text sentence by sentence
)

# For LLM-based segmentation with API calls, you might want to process in chunks
llm_segmenter = create_segmenter(
    backend="llm",
    strategy="chunk",    # Process text in chunks
    chunk_size=1000,     # Size of each chunk in characters
    api_key="your-key",
    model="gpt-3.5-turbo"
    endpoint="https://api.openai.com/v1"
)
```

The available strategies are:

| Strategy | Description | Best Use Cases |
|----------|-------------|----------------|
| `whole` | Process the entire text at once | Small to medium texts, when you need context across the entire document |
| `line` | Split by line breaks and process each line separately | Large documents, log files, or structured text with natural line breaks |
| `sentence` | Split into sentences and process each sentence | NLP tasks like word2vec that need sentence boundaries, or when sentence context is important |
| `chunk` | Split into fixed-size chunks and process each chunk | Very large documents, or when using API-based backends like LLM to avoid token limits |

Choosing the right strategy depends on:
- **Performance considerations**: Batched processing can be much faster
- **Memory constraints**: Processing large documents in smaller units reduces memory usage
- **API costs**: For LLM backends, processing in chunks can reduce API costs
- **Context requirements**: Some tasks need sentence boundaries preserved

### Creating a Segmenter

The `create_segmenter()` function is the main entry point for creating segmentation tools:

```python
from qhchina.preprocessing.segmentation import create_segmenter

# Create a segmenter with spaCy backend
segmenter = create_segmenter(
    backend="spacy",                    # Segmentation backend
    model_name="zh_core_web_sm",        # Lighter spaCy model
    batch_size=100,                     # Batch size for processing
    strategy="chunk",                   # Process sentence by sentence
    chunk_size=512,                     # Chunk size (for "chunk" strategy)
    filters={
        "min_word_length": 2,           # Minimum token length to include
        "excluded_pos": ["NUM", "SYM"], # POS tags to exclude
        "stopwords": ["的", "了"]       # Stopwords to exclude
    }
)

# Or create a segmenter with Jieba backend
jieba_segmenter = create_segmenter(
    backend="jieba",                    # Use Jieba backend
    pos_tagging=True,                   # Enable POS tagging
    strategy="line",                    # Process line by line
    filters={
        "min_word_length": 2,           # Minimum token length to include
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
  - Efficient batch processing using `nlp.pipe()` internally

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
  - Using "chunk" or "sentence" strategy is recommended to reduce API costs

### SpacySegmenter

The `SpacySegmenter` class provides Chinese text segmentation using spaCy models:

```python
from qhchina.preprocessing.segmentation import SpacySegmenter

# Create a spaCy-based segmenter
segmenter = SpacySegmenter(
    model_name="zh_core_web_sm",        # spaCy model to use
    disable=["ner", "lemmatizer"],      # Disable components for speed
    batch_size=100,                     # Batch size for processing
    user_dict=["量子物理", "深度学习"], # Custom user dictionary
    strategy="sentence",                # Process sentence by sentence
    filters={
        "min_word_length": 2,                # Min token length to keep
        "excluded_pos": ["NUM", "SYM", "SPACE"],  # POS tags to exclude
        "stopwords": ["的", "了"]       # Stopwords to exclude 
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
    strategy="line",                    # Process line by line
    filters={
        "min_word_length": 2,                # Min token length to keep
        "excluded_pos": ["m", "x"],     # POS tags to exclude (Jieba's POS tags)
        "stopwords": ["的", "了"],      # Words to exclude
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
    strategy="chunk",                   # Process in fixed-size chunks
    chunk_size=512,                     # Max sequence length for BERT
    filters={
        "min_word_length": 2,                # Min token length to keep
        "stopwords": ["的", "了"],      # Words to exclude
    }
)

# Segment text
text = "量子纠缠实验验证了非局域性原理。"
tokens = segmenter.segment(text)
print(tokens)  # Output: [['量子', '纠缠', '实验', '验证', '了', '非局域性', '原理', '。']]
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
    model="gpt-3.5-turbo",              # Model to use
    endpoint="https://api.openai.com/v1", # API endpoint URL - required parameter
    system_message="你是语言学专家。",   # System message for better segmentation
    temperature=0.1,                    # Lower temperature for more consistent results
    strategy="chunk",                   # Process in chunks to reduce API costs
    chunk_size=1000,                    # Size of each chunk in characters
    filters={
        "min_word_length": 1,           # Min token length to keep
        "stopwords": ["的", "了"]       # Words to exclude
    }
)

# Segment text
text = "脑机接口技术将改变人类与数字世界的交互方式。"
tokens = segmenter.segment(text)
print(tokens)  # Output: [['脑机', '接口', '技术', '将', '改变', '人类', '与', '数字', '世界', '的', '交互', '方式', '。']]

# Process multiple texts one by one
texts = ["虚拟现实创造沉浸式体验", "基因编辑技术引发伦理讨论"]
results = []
for text in texts:
    results.append(segmenter.segment(text))
print(results)
```

#### Custom Prompts

You can customize the segmentation prompt for different segmentation styles:

```python
# Custom prompt for academic-style segmentation
custom_prompt = """
请将以下中文文本分词，按照学术标准分词。请用JSON格式回答。

示例:
输入: "深度学习模型识别复杂图像"
输出: ["深度学习", "模型", "识别", "复杂", "图像"]

输入: "{text}"
输出:
"""

segmenter = LLMSegmenter(
    api_key="your-openai-api-key",
    model="gpt-4",
    endpoint="https://api.openai.com/v1", # API endpoint URL - required parameter
    prompt=custom_prompt,
    temperature=0,
    strategy="sentence"  # Process sentence by sentence
)
```

## Filtering Options

All segmenters support filtering options that can be passed during initialization:

| Filter | Description |
|--------|-------------|
| `min_word_length` | Minimum length of tokens to include (default: 1) |
| `excluded_pos` | Set of POS tags to exclude (requires POS tagging support) |
| `stopwords` | List of words to exclude from results |

## Common Workflows

### Basic Text Processing Pipeline

```python
from qhchina.preprocessing.segmentation import create_segmenter
from qhchina.helpers import load_texts, load_stopwords

# Load stopwords
stopwords = load_stopwords("zh_sim")

# Create segmenter with filters and strategy
segmenter = create_segmenter(
    backend="jieba",           # Using Jieba for faster processing
    strategy="sentence",       # Process sentence by sentence
    filters={
        "stopwords": stopwords, 
        "min_word_length": 2
    }
)

# Load and process texts
raw_texts = load_texts(["path/to/file1.txt", "path/to/file2.txt"])
processed_sentences = []

for text in raw_texts:
    # Segment text based on the strategy
    sentences = segmenter.segment(text)
    processed_sentences.extend(sentences)
    
# Now processed_documents is ready for analytics tasks
```

### Integration with Analytics

```python
from qhchina.preprocessing.segmentation import create_segmenter
from qhchina.analytics.topicmodels import LDAGibbsSampler

# Create segmenter with sentence strategy for word2vec or topic modeling
segmenter = create_segmenter(backend="spacy", strategy="sentence")

# Process text
text = """
宇宙起源理论存在多种可能性。暗物质构成宇宙的大部分质量。
量子力学和相对论难以统一。人类意识的本质仍是未解之谜。
"""
sentences = segmenter.segment(text)  # Returns a list of tokenized sentences

# Use processed text in analytics
lda = LDAGibbsSampler(n_topics=5)
lda.fit(sentences)

# Get topics
topics = lda.get_topics(n_words=5)
for i, topic in enumerate(topics):
    print(f"Topic {i}: {[word for word, _ in topic]}")
```

## Using Original Libraries Directly

### Using spaCy Directly

While qhChina's segmentation wrappers provide convenience and enhanced features, you can also use the underlying libraries directly. Here's how to use spaCy directly for Chinese text segmentation:

```python
import spacy
import re

# Download the Chinese model if not already installed
try:
    nlp = spacy.load("zh_core_web_sm")
except OSError:
    spacy.cli.download("zh_core_web_sm")
    nlp = spacy.load("zh_core_web_sm")

# Define a function to split text into sentences
def split_into_sentences(text, pattern=r"([。！？\.!?……]+)"):
    # Split by sentence-ending punctuation, but keep the punctuation
    raw_splits = re.split(pattern, text)
    
    # Combine sentence content with its ending punctuation
    sentences = []
    i = 0
    while i < len(raw_splits):
        if i + 1 < len(raw_splits) and re.match(pattern, raw_splits[i+1]):
            sentences.append(raw_splits[i] + raw_splits[i+1])
            i += 2
        else:
            if raw_splits[i].strip():
                sentences.append(raw_splits[i])
            i += 1
    
    return sentences

# Process text with different strategies
text = """量子计算将改变密码学的未来。
人工智能正在革新医疗诊断技术。
数据科学帮助解决复杂社会问题。"""

# Strategy 1: Process whole text at once
doc = nlp(text)
tokens_whole = [token.text for token in doc if not token.is_space]
print("Whole text processing:", tokens_whole)

# Strategy 2: Process line by line
lines = [line.strip() for line in text.split('\n') if line.strip()]
tokens_by_line = []
for line in lines:
    doc = nlp(line)
    tokens_by_line.append([token.text for token in doc if not token.is_space])
print("Line-by-line processing:", tokens_by_line)

# Strategy 3: Process sentence by sentence
sentences = split_into_sentences(text)
tokens_by_sentence = []
for sentence in sentences:
    doc = nlp(sentence)
    tokens_by_sentence.append([token.text for token in doc if not token.is_space])
print("Sentence-by-sentence processing:", tokens_by_sentence)

# Strategy 4: Batch processing with nlp.pipe() for efficiency
batch_texts = ["量子计算研究进展", "自然语言处理应用", "机器学习模型评估"]
tokens_batch = []
for doc in nlp.pipe(batch_texts, batch_size=50):
    tokens_batch.append([token.text for token in doc if not token.is_space])
print("Batch processing:", tokens_batch)

# Advanced: Filter by POS and length
filtered_tokens = []
stopwords = {"的", "了", "和"}
for doc in nlp.pipe(batch_texts):
    doc_tokens = [token.text for token in doc 
                  if not token.is_space 
                  and token.pos_ not in {"PUNCT", "NUM", "SYM"} 
                  and len(token.text) >= 2
                  and token.text not in stopwords]
    filtered_tokens.append(doc_tokens)
print("Filtered tokens:", filtered_tokens)
```

### Using Jieba Directly

Similarly, you can use the Jieba library directly for word segmentation:

```python
import jieba
import jieba.posseg as pseg
import re

# Optional: Add custom words to the dictionary
jieba.add_word("量子计算")
jieba.add_word("人工智能")

# Define a function to split text into sentences (same as above)
def split_into_sentences(text, pattern=r"([。！？\.!?……]+)"):
    raw_splits = re.split(pattern, text)
    sentences = []
    i = 0
    while i < len(raw_splits):
        if i + 1 < len(raw_splits) and re.match(pattern, raw_splits[i+1]):
            sentences.append(raw_splits[i] + raw_splits[i+1])
            i += 2
        else:
            if raw_splits[i].strip():
                sentences.append(raw_splits[i])
            i += 1
    return sentences

# Process text with different strategies
text = """量子计算将改变密码学的未来。
人工智能正在革新医疗诊断技术。
数据科学帮助解决复杂社会问题。"""

# Strategy 1: Process whole text at once
tokens_whole = list(jieba.cut(text))
print("Whole text processing:", tokens_whole)

# Strategy 2: Process line by line
lines = [line.strip() for line in text.split('\n') if line.strip()]
tokens_by_line = []
for line in lines:
    tokens_by_line.append(list(jieba.cut(line)))
print("Line-by-line processing:", tokens_by_line)

# Strategy 3: Process sentence by sentence
sentences = split_into_sentences(text)
tokens_by_sentence = []
for sentence in sentences:
    tokens_by_sentence.append(list(jieba.cut(sentence)))
print("Sentence-by-sentence processing:", tokens_by_sentence)

# With POS tagging
pos_tagged = []
stopwords = {"的", "了", "和"}
excluded_pos = {"x", "m", "r"}  # Exclude certain POS tags
for sentence in sentences:
    # Get tokens with POS tags
    tokens = pseg.cut(sentence)
    # Filter by POS and length
    filtered = [word for word, flag in tokens 
                if len(word) >= 2 
                and word not in stopwords
                and flag not in excluded_pos]
    pos_tagged.append(filtered)
print("POS-filtered tokens:", pos_tagged)

# Load a user dictionary from file
# jieba.load_userdict("path/to/dict.txt")
```

These examples demonstrate how to use spaCy and Jieba directly, but they require more manual coding compared to qhChina's wrappers, especially for handling different processing strategies, filtering, and batching.