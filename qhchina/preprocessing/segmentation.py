from typing import List, Dict, Any, Union, Optional, Set
from tqdm.auto import tqdm
import importlib
import re
import json
from datetime import datetime

class SegmentationWrapper:
    """Base segmentation wrapper class that can be extended for different segmentation tools."""
    
    def __init__(self, strategy: str = "whole", chunk_size: int = 512, filters: Dict[str, Any] = None, 
                 sentence_end_pattern: str = r"([。！？\.!?……]+)"):
        """Initialize the segmentation wrapper.
        
        Args:
            strategy (default "whole"): Strategy to process texts ['line', 'sentence', 'chunk', 'whole']
            chunk_size: Size of chunks when using 'chunk' strategy
            filters: Dictionary of filters to apply during segmentation
                - stopwords: List or set of stopwords to exclude (converted to set internally)
                - min_length: Minimum length of tokens to include (default 1)
                - excluded_pos: List or set of POS tags to exclude (converted to set internally)
            sentence_end_pattern: Regular expression pattern for sentence endings (default: Chinese and English punctuation)
        """
        self.strategy = strategy.strip().lower()
        if self.strategy not in ["line", "sentence", "chunk", "whole"]:
            raise ValueError(f"Invalid segmentation strategy: {strategy}. Must be one of: line, sentence, chunk, whole")
        
        self.chunk_size = chunk_size
        self.filters = filters or {}
        self.filters.setdefault('stopwords', set())
        if not isinstance(self.filters['stopwords'], set):
            self.filters['stopwords'] = set(self.filters['stopwords'])
        self.filters.setdefault('min_length', 1)
        self.filters.setdefault('excluded_pos', set())
        if not isinstance(self.filters['excluded_pos'], set):
            self.filters['excluded_pos'] = set(self.filters['excluded_pos'])
        self.sentence_end_pattern = sentence_end_pattern
    
    def segment(self, text: str) -> Union[List[str], List[List[str]]]:
        """Segment text into tokens based on the selected strategy.
        
        Args:
            text: Text to segment
            
        Returns:
            If strategy is 'whole': A single list of tokens
            If strategy is 'line', 'sentence', or 'chunk': A list of lists, where each inner list
            contains tokens for a line, sentence, or chunk respectively
        """
        # Split text based on the strategy
        units = self._split_text_by_strategy(text)
        
        # Process all units
        processed_results = self._process_all_texts(units)
        
        # For 'whole' strategy, merge all results into a single list
        if self.strategy == "whole" and processed_results:
            return processed_results[0]
        
        return processed_results
    
    def _split_text_by_strategy(self, text: str) -> List[str]:
        """Split text based on the selected strategy.
        
        Args:
            text: The text to split
            
        Returns:
            List of text units (lines, sentences, chunks, or whole text)
        """
        if self.strategy == "line":
            return self._split_into_lines(text)
        elif self.strategy == "sentence":
            return self._split_into_sentences(text)
        elif self.strategy == "chunk":
            return self._split_into_chunks(text, self.chunk_size)
        elif self.strategy == "whole":
            return [text] if text.strip() else []
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}")
    
    def _split_into_lines(self, text: str) -> List[str]:
        """Split text into non-empty lines.
        
        Args:
            text: Text to split into lines
            
        Returns:
            List of non-empty lines
        """
        return [line.strip() for line in text.split('\n') if line.strip()]
    
    def _split_into_chunks(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks of specified size.
        
        Args:
            text: Text to split into chunks
            chunk_size: Maximum size of each chunk
            
        Returns:
            List of text chunks
        """
        # Return empty list for empty text
        if not text:
            return []
            
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i + chunk_size])
            
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences.
        
        Args:
            text: Text to split into sentences
            
        Returns:
            List of sentences
        """
        # Simple Chinese sentence-ending punctuation pattern
        sentence_end_pattern = self.sentence_end_pattern
        
        # Split by sentence-ending punctuation, but keep the punctuation
        raw_splits = re.split(sentence_end_pattern, text)
        
        # Combine sentence content with its ending punctuation
        sentences = []
        i = 0
        while i < len(raw_splits):
            if i + 1 < len(raw_splits) and re.match(sentence_end_pattern, raw_splits[i+1]):
                sentences.append(raw_splits[i] + raw_splits[i+1])
                i += 2
            else:
                if raw_splits[i].strip():
                    sentences.append(raw_splits[i])
                i += 1
        
        # If no sentences were found, treat the whole text as one sentence
        if not sentences and text.strip():
            sentences = [text]
        
        return sentences
    
    def _process_all_texts(self, texts: List[str]) -> List[List[str]]:
        """Process all text units and return results.
        
        Args:
            texts: List of all text units to process
            
        Returns:
            List of processed results for each text unit
            
        Note:
            This method should be implemented by subclasses
        """
        raise NotImplementedError("This method should be implemented by subclasses")


class SpacySegmenter(SegmentationWrapper):
    """Segmentation wrapper for spaCy models."""
    
    def __init__(self, model_name: str = "zh_core_web_lg", 
                 disable: Optional[List[str]] = None,
                 batch_size: int = 200,
                 user_dict: Union[List[str], str] = None,
                 strategy: str = "whole", 
                 chunk_size: int = 512,
                 filters: Dict[str, Any] = None,
                 sentence_end_pattern: str = r"([。！？\.!?……]+)"):
        """Initialize the spaCy segmenter.
        
        Args:
            model_name: Name of the spaCy model to use
            disable: List of pipeline components to disable for better performance; default setting is ["ner", "lemmatizer"]
            batch_size: Batch size for processing multiple texts
            user_dict: Custom user dictionary - either a list of words or path to a dictionary file
            strategy: Strategy to process texts ['line', 'sentence', 'chunk', 'whole']
            chunk_size: Size of chunks when using 'chunk' strategy
            filters: Dictionary of filters to apply during segmentation
                - min_length: Minimum length of tokens to include (default 1)
                - excluded_pos: Set of POS tags to exclude from token outputs
                - stopwords: Set of stopwords to exclude
            sentence_end_pattern: Regular expression pattern for sentence endings
        """
        super().__init__(strategy=strategy, chunk_size=chunk_size, filters=filters, 
                         sentence_end_pattern=sentence_end_pattern)
        self.model_name = model_name
        self.disable = disable or []
        self.batch_size = batch_size
        self.user_dict = user_dict
        
        # Try to load the model, download if needed
        try:
            import spacy
        except ImportError:
            raise ImportError("spacy is not installed. Please install it with 'pip install spacy'")
        
        try:
            self.nlp = spacy.load(model_name, disable=self.disable)
        except OSError:
            # Model not found, try to download it
            try:
                if importlib.util.find_spec("spacy.cli") is not None:
                    spacy.cli.download(model_name)
                else:
                    # Manual import as fallback
                    from spacy.cli import download
                    download(model_name)
                # Load the model after downloading
                self.nlp = spacy.load(model_name, disable=self.disable)
                print(f"Model {model_name} successfully downloaded and loaded.")
            except Exception as e:
                raise ImportError(
                    f"Could not download model {model_name}. Error: {str(e)}. "
                    f"Please install it manually with 'python -m spacy download {model_name}'")
        
        # Update user dictionary if provided
        if self.user_dict is not None:
            self._update_user_dict()
    
    def _update_user_dict(self):
        """Update the tokenizer's user dictionary."""
        # Check if the model supports pkuseg user dictionary update
        if hasattr(self.nlp.tokenizer, 'pkuseg_update_user_dict'):
            try:
                # If user_dict is a file path
                if isinstance(self.user_dict, str):
                    try:
                        with open(self.user_dict, 'r', encoding='utf-8') as f:
                            words = [line.strip() for line in f if line.strip()]
                        self.nlp.tokenizer.pkuseg_update_user_dict(words)
                        print(f"Loaded user dictionary from file: {self.user_dict}")
                    except Exception as e:
                        print(f"Failed to load user dictionary from file: {str(e)}")
                # If user_dict is a list of words
                elif isinstance(self.user_dict, list):
                    self.nlp.tokenizer.pkuseg_update_user_dict(self.user_dict)
                    print(f"Updated user dictionary with {len(self.user_dict)} words")
                else:
                    print(f"Unsupported user_dict type: {type(self.user_dict)}. Expected str or list.")
            except Exception as e:
                print(f"Failed to update user dictionary: {str(e)}")
        else:
            print("Warning: This spaCy model's tokenizer does not support pkuseg_update_user_dict")
    
    def _filter_tokens(self, tokens):
        """Filter tokens based on excluded POS tags and minimum length."""
        min_length = self.filters.get('min_length', 1)
        excluded_pos = self.filters.get('excluded_pos', [])
        stopwords = set(self.filters.get('stopwords', []))
        return [token for token in tokens 
                if token.pos_ not in excluded_pos 
                and len(token.text) >= min_length
                and token.text not in stopwords]
    
    def _process_all_texts(self, texts: List[str]) -> List[List[str]]:
        """Process all texts with spaCy's pipe and return results.
        
        Args:
            texts: List of all text units to process
            
        Returns:
            List of lists of tokens, one list per text unit
        """
        results = []
        
        # Process each doc and add results
        for doc in tqdm(self.nlp.pipe(texts, batch_size=self.batch_size), 
                         total=len(texts), desc="Segmenting with spaCy"):
            # Get filtered tokens for the doc
            filtered_tokens = [token.text for token in self._filter_tokens(doc)]
            
            # Add to results
            results.append(filtered_tokens)
        
        return results


class JiebaSegmenter(SegmentationWrapper):
    """Segmentation wrapper for Jieba Chinese text segmentation."""
    
    def __init__(self, 
                 user_dict_path: str = None,
                 pos_tagging: bool = False,
                 strategy: str = "whole",
                 chunk_size: int = 512,
                 filters: Dict[str, Any] = None,
                 sentence_end_pattern: str = r"([。！？\.!?……]+)"):
        """Initialize the Jieba segmenter.
        
        Args:
            user_dict_path: Path to a user dictionary file for Jieba
            pos_tagging: Whether to include POS tagging in segmentation
            strategy: Strategy to process texts ['line', 'sentence', 'chunk', 'whole']
            chunk_size: Size of chunks when using 'chunk' strategy
            filters: Dictionary of filters to apply during segmentation
                - min_length: Minimum length of tokens to include (default 1)
                - excluded_pos: List of POS tags to exclude (if pos_tagging is True)
                - stopwords: Set of stopwords to exclude
            sentence_end_pattern: Regular expression pattern for sentence endings
        """
        super().__init__(strategy=strategy, chunk_size=chunk_size, filters=filters,
                         sentence_end_pattern=sentence_end_pattern)
        self.pos_tagging = pos_tagging
        
        # Try to import jieba
        try:
            import jieba
            import jieba.posseg as pseg
        except ImportError:
            raise ImportError("jieba is not installed. Please install it with 'pip install jieba'")
        
        self.jieba = jieba
        self.pseg = pseg
        
        # Load user dictionary if provided
        if user_dict_path:
            try:
                self.jieba.load_userdict(user_dict_path)
                print(f"Loaded user dictionary from {user_dict_path}")
            except Exception as e:
                print(f"Failed to load user dictionary: {str(e)}")
    
    def _filter_tokens(self, tokens) -> List[str]:
        """Filter tokens based on filters."""
        min_length = self.filters.get('min_length', 1)
        stopwords = set(self.filters.get('stopwords', []))
        
        # If POS tagging is enabled and we have tokens as (word, flag) tuples
        if self.pos_tagging:
            excluded_pos = set(self.filters.get('excluded_pos', []))
            return [word for word, flag in tokens 
                    if len(word) >= min_length 
                    and word not in stopwords
                    and flag not in excluded_pos]
        else:
            return [token for token in tokens 
                    if len(token) >= min_length 
                    and token not in stopwords]
    
    def _process_all_texts(self, texts: List[str]) -> List[List[str]]:
        """Process all text units with Jieba and return results.
        
        Args:
            texts: List of all text units to process
            
        Returns:
            List of lists of tokens, one list per text unit
        """
        results = []
        for text_to_process in tqdm(texts, desc="Segmenting with Jieba"):
            # Skip empty text
            if not text_to_process.strip():
                results.append([])
                continue
                
            if self.pos_tagging:
                tokens = list(self.pseg.cut(text_to_process))
            else:
                tokens = list(self.jieba.cut(text_to_process))

            # Filter tokens
            filtered_tokens = self._filter_tokens(tokens)
            
            # Add to results
            results.append(filtered_tokens)
        
        return results


class BertSegmenter(SegmentationWrapper):
    """Segmentation wrapper for BERT-based Chinese word segmentation."""
    
    # Predefined tagging schemes
    TAGGING_SCHEMES = {
        "be": ["B", "E"],  # B: beginning of word, E: end of word
        "bme": ["B", "M", "E"],  # B: beginning, M: middle, E: end
        "bmes": ["B", "M", "E", "S"]  # B: beginning, M: middle, E: end, S: single
    }
    
    def __init__(self, 
                 model_name: str = None,
                 model = None,
                 tokenizer = None,
                 tagging_scheme: Union[str, List[str]] = "be",
                 batch_size: int = 32,
                 device: Optional[str] = None,
                 remove_special_tokens: bool = True,
                 max_sequence_length: int = 512,
                 strategy: str = "whole",
                 chunk_size: int = 512,
                 filters: Dict[str, Any] = None,
                 sentence_end_pattern: str = r"([。！？\.!?……]+)"):
        """Initialize the BERT segmenter.
        
        Args:
            model_name: Name of the pre-trained BERT model to load (optional if model and tokenizer are provided)
            model: Pre-initialized model instance (optional if model_name is provided)
            tokenizer: Pre-initialized tokenizer instance (optional if model_name is provided)
            tagging_scheme: Either a string ('be', 'bmes') or a list of tags in their exact order (e.g. ["B", "E"]).
                           When a list is provided, the order of tags matters as it maps to prediction indices.
                           For example, ["B", "E"] is interpreted as [0, 1] in the prediction.
            batch_size: Batch size for processing
            device: Device to use ('cpu', 'cuda', etc.)
            remove_special_tokens: Whether to remove special tokens (CLS, SEP) from output, default is True, which works for BERT-based models.
            max_sequence_length: Maximum sequence length for BERT models (default 512); if the text is longer than this, it will be split into chunks.
            strategy: Strategy to process texts ['line', 'sentence', 'chunk', 'whole']
            chunk_size: Size of chunks when using 'chunk' strategy
            filters: Dictionary of filters to apply during segmentation
                - min_length: Minimum length of tokens to include (default 1)
                - excluded_pos: Set of POS tags to exclude from token outputs
                - stopwords: Set of stopwords to exclude
            sentence_end_pattern: Regular expression pattern for sentence endings
        """
        # Use max_sequence_length as chunk_size if not provided separately
        if not chunk_size and strategy == "chunk":
            chunk_size = max_sequence_length
        
        super().__init__(strategy=strategy, chunk_size=chunk_size, filters=filters,
                         sentence_end_pattern=sentence_end_pattern)
        self.batch_size = batch_size
        self.remove_special_tokens = remove_special_tokens
        self.max_sequence_length = max_sequence_length
        
        # Validate that either model_name or both model and tokenizer are provided
        if model_name is None and (model is None or tokenizer is None):
            raise ValueError("Either model_name or both model and tokenizer must be provided")
        
        # Handle tagging scheme - can be a string or a list
        if isinstance(tagging_scheme, str):
            # String-based predefined scheme
            if tagging_scheme.lower() not in self.TAGGING_SCHEMES:
                raise ValueError(f"Unsupported tagging scheme: {tagging_scheme}. "
                               f"Supported schemes: {list(self.TAGGING_SCHEMES.keys())}")
            self.tagging_scheme_name = tagging_scheme.lower()
            self.labels = self.TAGGING_SCHEMES[self.tagging_scheme_name]
        elif isinstance(tagging_scheme, list):
            # Direct list of tags
            if not tagging_scheme:
                raise ValueError("Tagging scheme list cannot be empty")
            self.tagging_scheme_name = "custom"
            self.labels = tagging_scheme
        else:
            raise ValueError("tagging_scheme must be either a string or a list of tags")
        
        # Try to import transformers
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForTokenClassification
            self.torch = torch
            self.AutoTokenizer = AutoTokenizer
            self.AutoModelForTokenClassification = AutoModelForTokenClassification
        except ImportError:
            raise ImportError("transformers and torch are not installed. "
                             "Please install them with 'pip install transformers torch'")
        
        # Set device
        if device is None:
            self.device = 'cuda' if self.torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Initialize model and tokenizer
        if model is not None and tokenizer is not None:
            # Use provided model and tokenizer
            self.model = model.to(self.device)
            self.tokenizer = tokenizer
            print(f"Using provided model and tokenizer on {self.device}")
        else:
            # Load model and tokenizer from pretrained
            try:
                self.tokenizer = self.AutoTokenizer.from_pretrained(model_name)
                self.model = self.AutoModelForTokenClassification.from_pretrained(
                    model_name, 
                    num_labels=len(self.labels)
                ).to(self.device)
                print(f"Loaded model {model_name} on {self.device}")
            except Exception as e:
                raise ImportError(f"Failed to load model {model_name}. Error: {str(e)}")
        
        self.model.eval()
        print(f"Using tagging scheme: {self.labels}")
    
    def _filter_words(self, words: List[str]) -> List[str]:
        """Filter words based on specified filters.
        
        Args:
            words: List of words to filter
            
        Returns:
            Filtered list of words
        """
        min_length = self.filters.get('min_length', 1)
        stopwords = set(self.filters.get('stopwords', []))
        
        return [word for word in words 
                if len(word) >= min_length 
                and word not in stopwords]
    
    def _predict_tags_batch(self, texts: List[str]) -> List[List[str]]:
        """Predict segmentation tags for each character in a batch of texts."""
        # Process each text to character level and store original lengths
        all_tokens = []
        original_lengths = []
        
        for text in texts:
            tokens = list(text)
            all_tokens.append(tokens)
            original_lengths.append(len(tokens))
        
        # Tokenize all texts at character level
        inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=self.max_sequence_length
        ).to(self.device)
        
        # Get predictions
        with self.torch.no_grad():
            outputs = self.model(**inputs)
            predictions = self.torch.argmax(outputs.logits, dim=2)
        
        # Process predictions back to tags for each text
        all_tags = []
        for pred, original_length in zip(predictions, original_lengths):
            # Skip special tokens like [CLS] and [SEP] if configured to do so
            if self.remove_special_tokens:
                # BERT tokenization adds [CLS] at start and [SEP] at end:
                # [CLS] char1 char2 ... charN [SEP]
                # So we only need positions in range (1, original length+1)
                pred_length = len(pred)
                end_idx = min(original_length + 1, pred_length)
                tags = [self.labels[p.item()] for p in pred[1:end_idx]]  # Skip [CLS], include all characters
            else:
                # Include special tokens - but still limit to the actual content length
                tags = [self.labels[p.item()] for p in pred[:original_length+1]]
            
            all_tags.append(tags)
        
        return all_tags
    
    def _predict_tags(self, text: str) -> List[str]:
        """Predict segmentation tags for each character in a single text."""
        return self._predict_tags_batch([text])[0]
    
    def _merge_tokens_by_tags(self, tokens: List[str], tags: List[str]) -> List[str]:
        """Merge tokens based on predicted tags."""
        words = []
        current_word = ""
        
        # BE tagging scheme
        if len(self.labels) == 2 and all(tag in self.labels for tag in tags):
            b_index = self.labels.index("B")
            e_index = self.labels.index("E")
            
            for token, tag in zip(tokens, tags):
                if tag == self.labels[b_index]:  # Beginning of a word
                    if current_word:
                        words.append(current_word)
                    current_word = token
                elif tag == self.labels[e_index]:  # End of a word
                    current_word += token
                    words.append(current_word)
                    current_word = ""
                else:  # Fallback for any other tag
                    if current_word:
                        current_word += token
                    else:
                        words.append(token)
            
            # Add the last word if it exists
            if current_word:
                words.append(current_word)
        
        # BME tagging scheme (3-tag scheme)
        elif len(self.labels) == 3 and all(tag in self.labels for tag in tags):
            b_index = self.labels.index("B")
            m_index = self.labels.index("M")
            e_index = self.labels.index("E")
            
            for token, tag in zip(tokens, tags):
                if tag == self.labels[b_index]:  # Beginning of a word
                    if current_word:
                        words.append(current_word)
                    current_word = token
                elif tag == self.labels[m_index]:  # Middle of a word
                    current_word += token
                elif tag == self.labels[e_index]:  # End of a word
                    current_word += token
                    words.append(current_word)
                    current_word = ""
                else:  # Fallback for any other tag
                    if current_word:
                        current_word += token
                    else:
                        words.append(token)
            
            # Add the last word if it exists
            if current_word:
                words.append(current_word)
        
        # BMES tagging scheme
        elif len(self.labels) == 4 and all(tag in self.labels for tag in tags):
            b_index = self.labels.index("B")
            m_index = self.labels.index("M")
            e_index = self.labels.index("E")
            s_index = self.labels.index("S")
            
            for token, tag in zip(tokens, tags):
                if tag == self.labels[b_index]:  # Beginning of a multi-character word
                    current_word = token
                elif tag == self.labels[m_index]:  # Middle of a multi-character word
                    current_word += token
                elif tag == self.labels[e_index]:  # End of a multi-character word
                    current_word += token
                    words.append(current_word)
                    current_word = ""
                elif tag == self.labels[s_index]:  # Single character word
                    words.append(token)
                else:  # Fallback for any other tag
                    if current_word:
                        current_word += token
                    else:
                        words.append(token)
            
            # Add the last word if it exists
            if current_word:
                words.append(current_word)
        
        return words
    
    def _process_all_texts(self, texts: List[str]) -> List[List[str]]:
        """Process all texts with BERT model and return results.
        
        Args:
            texts: List of all text units to process
            
        Returns:
            List of lists of tokens, one list per text unit
        """
        # Initialize results list
        results = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
                
            # Get tokens and tags for each text in the batch
            batch_tokens = [list(text) for text in batch_texts]
            batch_tags = self._predict_tags_batch(batch_texts)
            
            # Process each text in the batch
            for tokens, tags in zip(batch_tokens, batch_tags):
                # Make sure tags and tokens match in length
                if len(tags) != len(tokens):
                    print(f"Warning: Tags and tokens length mismatch. Tags: {len(tags)}, Tokens: {len(tokens)}")
                    results.append([])  # Add empty list for this entry
                    continue
                    
                words = self._merge_tokens_by_tags(tokens, tags)
                
                # Apply filters
                filtered_words = self._filter_words(words)
                
                # Add to results
                results.append(filtered_words)
        
        return results


class LLMSegmenter(SegmentationWrapper):
    """Segmentation wrapper using Language Model APIs like OpenAI."""
    
    DEFAULT_PROMPT = """
    请将以下中文文本分词。请用JSON格式回答。
    
    示例:
    输入: "今天天气真好，我们去散步吧！"
    输出: ["今天", "天气", "真", "好", "，", "我们", "去", "散步", "吧", "！"]
    
    输入: "{text}"
    输出:
    """
    
    def __init__(self, 
                 api_key: str,
                 model: str,
                 endpoint: str,
                 prompt: str = None,
                 system_message: str = None,
                 temperature: float = 1,
                 max_tokens: int = 2048,
                 strategy: str = "whole",
                 chunk_size: int = 512,
                 filters: Dict[str, Any] = None,
                 sentence_end_pattern: str = r"([。！？\.!?……]+)"):
        """Initialize the LLM segmenter.
        
        Args:
            api_key: API key for the language model service
            model: Model name to use
            endpoint: API endpoint URL
            prompt: Custom prompt template with {text} placeholder (if None, uses DEFAULT_PROMPT)
            system_message: Optional system message to prepend to API calls
            temperature: Temperature for model sampling (lower for more deterministic output)
            max_tokens: Maximum tokens in the response
            strategy: Strategy to process texts ['line', 'sentence', 'chunk', 'whole']
            chunk_size: Size of chunks when using 'chunk' strategy
            filters: Dictionary of filters to apply during segmentation
                - min_length: Minimum length of tokens to include (default 1)
                - excluded_pos: Set of POS tags to exclude from token outputs
                - stopwords: Set of stopwords to exclude
            sentence_end_pattern: Regular expression pattern for sentence endings
        """
        super().__init__(strategy=strategy, chunk_size=chunk_size, filters=filters,
                         sentence_end_pattern=sentence_end_pattern)
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint
        self.prompt = prompt or self.DEFAULT_PROMPT
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Try to import OpenAI
        try:
            import openai
        except ImportError:
            raise ImportError("openai is not installed. Please install it with 'pip install openai'")
        
        # Configure OpenAI client
        if endpoint:
            # Custom API endpoint
            self.client = openai.OpenAI(
                api_key=api_key,
                base_url=endpoint
            )
        else:
            # Default OpenAI endpoint
            self.client = openai.OpenAI(api_key=api_key)
    
    def _call_llm_api(self, text: str) -> List[str]:
        """Call the LLM API with the provided text and parse the response as a list of tokens."""
        prompt_text = self.prompt.format(text=text)
        
        try:
            # Prepare the messages
            messages = []
            
            # Add system message if provided
            if self.system_message:
                messages.append({"role": "system", "content": self.system_message})
                
            # Add user message with the prompt
            messages.append({"role": "user", "content": prompt_text})
            
            # Call the API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            response_text = response.choices[0].message.content
            
            # Try to parse as JSON
            try:
                # First check if response is already a list
                if response_text.strip().startswith('[') and response_text.strip().endswith(']'):
                    try:
                        tokens = json.loads(response_text)
                        if isinstance(tokens, list):
                            return tokens
                    except json.JSONDecodeError as je:
                        print(f"Warning: Response looks like a list but isn't valid JSON: {str(je)}")
                        print(f"Response text (first 100 chars): {response_text[:100]}...")
                
                # Try to extract JSON structure from the response
                try:
                    parsed_json = json.loads(response_text)
                    
                    # Check for common API response patterns
                    if isinstance(parsed_json, list):
                        return parsed_json
                    elif 'tokens' in parsed_json:
                        return parsed_json['tokens']
                    elif 'words' in parsed_json:
                        return parsed_json['words']
                    elif 'segments' in parsed_json:
                        return parsed_json['segments']
                    elif 'result' in parsed_json:
                        return parsed_json['result']
                    elif 'results' in parsed_json:
                        return parsed_json['results']
                    else:
                        # Just return the first list found in the JSON
                        for value in parsed_json.values():
                            if isinstance(value, list) and len(value) > 0:
                                return value
                        
                        # If we didn't find any list, log this unusual response
                        print(f"Warning: No list found in JSON response: {parsed_json}")
                        # Fallback to raw tokens if no list found
                        return []
                except json.JSONDecodeError as je:
                    # Show detailed error for debugging
                    print(f"JSON Decode Error: {str(je)}")
                    print(f"Response text (first 100 chars): {response_text[:100]}...")
                    return []
                    
            except Exception as e:
                print(f"Error parsing API response: {str(e)}")
                print(f"Response text (first 100 chars): {response_text[:100]}...")
                return []
                
        except Exception as e:
            print(f"Error calling LLM API: {str(e)}")
            return []
    
    def _filter_tokens(self, tokens: List[str]) -> List[str]:
        """Apply filters to the tokens."""
        min_length = self.filters.get('min_length', 1)
        stopwords = set(self.filters.get('stopwords', []))
        
        return [token for token in tokens 
                if len(token) >= min_length 
                and token not in stopwords]
    
    def _process_all_texts(self, texts: List[str]) -> List[List[str]]:
        """Process all text units with LLM API and return results.
        
        Args:
            texts: List of all text units to process
            
        Returns:
            List of lists of tokens, one list per text unit
        """
        # Process each text unit one by one (no batching for API calls)
        results = []
        for text_to_process in tqdm(texts, desc=f"Segmenting with {self.model}"):
            
            # Call the LLM API for this text unit
            tokens = self._call_llm_api(text_to_process)
            filtered_tokens = self._filter_tokens(tokens)
            
            # Add the tokens to results
            results.append(filtered_tokens)
                
        return results

# Factory function to create appropriate segmenter based on the backend
def create_segmenter(backend: str = "spacy", strategy: str = "whole", chunk_size: int = 512, 
                  sentence_end_pattern: str = r"([。！？\.!?……]+)", **kwargs) -> SegmentationWrapper:
    """Create a segmenter based on the specified backend.
    
    Args:
        backend: The segmentation backend to use ('spacy', 'jieba', 'bert', 'llm', etc.)
        strategy: Strategy to process texts ['line', 'sentence', 'chunk', 'whole']
        chunk_size: Size of chunks when using 'chunk' strategy
        sentence_end_pattern: Regular expression pattern for sentence endings (default: Chinese and English punctuation)
        **kwargs: Additional arguments to pass to the segmenter constructor
            - filters: Dictionary of filters to apply during segmentation
                - min_length: Minimum length of tokens to include (default 1)
                - stopwords: Set of stopwords to exclude
                - excluded_pos: Set of POS tags to exclude (for backends that support POS tagging)
            - Other backend-specific arguments
        
    Returns:
        An instance of a SegmentationWrapper subclass
        
    Raises:
        ValueError: If the specified backend is not supported
    """
    # Add strategy and chunk_size to kwargs
    kwargs['strategy'] = strategy
    kwargs['chunk_size'] = chunk_size
    kwargs['sentence_end_pattern'] = sentence_end_pattern
    
    if backend.lower() == "spacy":
        return SpacySegmenter(**kwargs)
    elif backend.lower() == "jieba":
        return JiebaSegmenter(**kwargs)
    elif backend.lower() == "bert":
        return BertSegmenter(**kwargs)
    elif backend.lower() == "llm":
        return LLMSegmenter(**kwargs)
    else:
        raise ValueError(f"Unsupported segmentation backend: {backend}")