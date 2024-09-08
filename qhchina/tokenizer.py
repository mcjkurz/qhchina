# qhchina/tokenizer.py
import re
import subprocess
import sys
import importlib.util

class Tokenizer:
    def __init__(self, backend="default", model_name="zh_core_web_lg", batch_size=10):
        """Initialize the tokenizer with the specified backend and model."""
        self.backend = backend.lower()
        self.batch_size = batch_size
        self.model_name = model_name
        self.nlp = None
        if self.backend != "default":
            self.setup_tokenizer()

    def setup_tokenizer(self):
        """Setup the tokenizer based on the backend (spacy, jieba, etc.)."""
        if self.backend == "spacy":
            self.setup_spacy()
        elif self.backend == "jieba":
            self.setup_jieba()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def setup_spacy(self):
        """Set up spaCy tokenizer, install and load the model if necessary."""
        try:
            spacy = importlib.util.find_spec("spacy")
            if spacy is None:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy"])
            import spacy

            # Check if the model is already loaded
            if self.nlp is None:
                try:
                    self.nlp = spacy.load(self.model_name)
                except OSError:
                    subprocess.check_call([sys.executable, "-m", "spacy", "download", self.model_name])
                    self.nlp = spacy.load(self.model_name)
        except Exception as e:
            print(f"Error setting up spaCy: {e}")

    def setup_jieba(self):
        """Set up jieba tokenizer."""
        try:
            jieba = importlib.util.find_spec("jieba")
            if jieba is None:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "jieba"])
            import jieba
            self.nlp = jieba
        except Exception as e:
            print(f"Error setting up jieba: {e}")

    def tokenize(self, text, delimiter=None, regex=False, disable=[], batch_size=None):
        """Tokenize text or a batch of texts using the selected backend."""
        if isinstance(text, list):  # Automatically detect batch mode
            if batch_size is None:
                batch_size = self.batch_size
            return self.tokenize_batch(texts=text, delimiter=delimiter, regex=regex, disable=disable, batch_size=batch_size)
        else:
            return self.tokenize_single(text, delimiter=delimiter, regex=regex, disable=disable)

    def tokenize_single(self, text, delimiter=None, regex=False, disable=[]):
        """Tokenize a single text, either using a backend or by splitting on a delimiter."""
        if self.backend == "spacy":
            return self.tokenize_with_spacy(text, disable=disable)
        elif self.backend == "jieba":
            return self.tokenize_with_jieba(text)
        else:
            return self.tokenize_with_split(text, delimiter=delimiter, regex=regex)

    def tokenize_batch(self, texts, delimiter=None, regex=False, disable=[], batch_size=10):
        """Tokenize a batch of texts, processing in chunks if needed."""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            if self.backend == "spacy":
                results.extend(self.tokenize_batch_with_spacy(batch, disable=disable))
            elif self.backend == "jieba":
                results.extend([self.tokenize_with_jieba(text) for text in batch])
            else:
                results.extend([self.tokenize_with_split(text, delimiter=delimiter, regex=regex) for text in batch])
        return results

    def tokenize_with_spacy(self, text, disable=[]):
        """Tokenize a single text using spaCy's model."""
        if self.nlp:
            doc = self.nlp(text, disable=disable)
            return [token.text for token in doc]
        else:
            raise Exception("spaCy model not loaded")

    def tokenize_batch_with_spacy(self, texts, disable=[]):
        """Tokenize a batch of texts using spaCy's 'pipe' method for efficiency."""
        if self.nlp:
            docs = self.nlp.pipe(texts, disable=disable)
            return [[token.text for token in doc] for doc in docs]
        else:
            raise Exception("spaCy model not loaded")

    def tokenize_with_jieba(self, text):
        """Tokenize using jieba."""
        if self.nlp:
            return list(self.nlp.cut(text))
        else:
            raise Exception("jieba tokenizer not available")

    def tokenize_with_split(self, text, delimiter=None, regex=False):
        """Tokenize using a custom delimiter or regex split."""
        if regex:
            if delimiter is None:
                delimiter = r"\s+"  # Default regex: split by any whitespace
            return re.split(delimiter, text)
        else:
            if delimiter is None:
                return list(text)
            return text.split(delimiter)

tokenizer = Tokenizer()
tokens = tokenizer.tokenize("这是我的文章")
print(tokens)

# Tokenizing by space
tokens = tokenizer.tokenize("This is a test", delimiter=" ")
print(tokens)

# Tokenizing a batch of texts with spaCy
texts = ["这是第一篇文章", "这是第二篇文章"]
tokenizer = Tokenizer(backend="spacy", model_name="zh_core_web_lg", batch_size=2)
tokens_batch = tokenizer.tokenize(texts, batch_size=2, disable=["ner", "parser"])
print(tokens_batch)

# Tokenizing a batch of texts using custom delimiter
texts = ["first text", "second text"]
tokenizer = Tokenizer()
tokens_batch = tokenizer.tokenize(texts, delimiter=" ", batch_size=1)
print(tokens_batch)