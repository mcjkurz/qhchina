from typing import List, Dict, Any, Union, Optional, Set
from tqdm.auto import tqdm
from qhchina.helpers.texts import split_into_chunks
import importlib
import re


class SegmentationWrapper:
    """Base segmentation wrapper class that can be extended for different segmentation tools."""
    
    def __init__(self, filters: Dict[str, Any] = None):
        """Initialize the segmentation wrapper.
        
        Args:
            filters: Dictionary of filters to apply during segmentation
        """
        self.filters = filters or {}
        self.filters.setdefault('stopwords', [])
        self.filters.setdefault('min_sentence_length', 1)
        self.filters.setdefault('min_token_length', 1)
        self.filters.setdefault('excluded_pos', [])
    
    def segment(self, text: Union[str, List[str]]) -> Union[List[str], List[List[str]]]:
        """Segment text(s) into tokens, removing unwanted tokens.
        
        Args:
            text: Single text or list of texts to segment
            
        Returns:
            A list of tokens for a single text, or a list of lists of tokens for multiple texts
        """
        raise NotImplementedError("This method should be implemented by subclasses")
    
    def segment_to_sentences(self, text: str) -> List[List[str]]:
        """Segment text into sentences, where each sentence is a list of tokens.
        
        The text is first split into passages (non-empty lines), then each 
        passage is processed separately, and sentences are extracted.
        
        Args:
            text: Text to segment
            
        Returns:
            A list of lists of tokens, where each inner list represents a tokenized sentence
        """
        raise NotImplementedError("This method should be implemented by subclasses")
        
    def _split_text_into_passages(self, text: str) -> List[str]:
        """Split text into passages (non-empty lines)."""
        return [line.strip() for line in text.split('\n') if line.strip()]


class SpacySegmenter(SegmentationWrapper):
    """Segmentation wrapper for spaCy models."""
    
    def __init__(self, model_name: str = "zh_core_web_lg", 
                 disable: Optional[List[str]] = ["ner", "lemmatizer"],
                 batch_size: int = 200,
                 user_dict: Union[List[str], str] = None,
                 filters: Dict[str, Any] = None):
        """Initialize the spaCy segmenter.
        
        Args:
            model_name: Name of the spaCy model to use
            disable: List of pipeline components to disable for better performance; default setting is ["ner", "lemmatizer"]
            batch_size: Batch size for processing multiple texts
            user_dict: Custom user dictionary - either a list of words or path to a dictionary file
            filters: Dictionary of filters to apply during segmentation
                - min_sentence_length: Minimum length of sentences to include (default 1)
                - min_token_length: Minimum length of tokens to include (default 1)
                - excluded_pos: Set of POS tags to exclude from token outputs (default: NUM, SYM, SPACE)
        """
        super().__init__(filters)
        self.model_name = model_name
        self.disable = disable
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
        min_length = self.filters.get('min_token_length', 1)
        excluded_pos = self.filters.get('excluded_pos', [])
        return [token for token in tokens 
                if token.pos_ not in excluded_pos and len(token.text) >= min_length]
    
    def segment(self, text: Union[str, List[str]]) -> Union[List[str], List[List[str]]]:
        """Segment text(s) into tokens, removing unwanted tokens.
        
        Args:
            text: Single text or list of texts to segment
            
        Returns:
            A list of tokens for a single text, or a list of lists of tokens for multiple texts
        """
        # Handle single text case
        if isinstance(text, str):
            # Check if the text is longer than the model's max length
            if len(text) > self.nlp.max_length:
                # Split text into chunks and process each chunk
                chunks = split_into_chunks(text, self.nlp.max_length)
                all_tokens = []
                
                for chunk in chunks:
                    doc = self.nlp(chunk)
                    tokens = [token.text for token in self._filter_tokens(doc)]
                    all_tokens.extend(tokens)
                
                return all_tokens
            else:
                # Process normally if text is within length limits
                doc = self.nlp(text)
                return [token.text for token in self._filter_tokens(doc)]
        
        # Handle multiple texts case with batching
        results = []
        for doc in tqdm(self.nlp.pipe(text, batch_size=self.batch_size), 
                       total=len(text)):
            tokens = [token.text for token in self._filter_tokens(doc)]
            results.append(tokens)
        
        return results
    
    def segment_to_sentences(self, text: str) -> List[List[str]]:
        """Segment text into sentences, where each sentence is a list of tokens.
        
        The text is first split into passages (non-empty lines), then each 
        passage is processed separately, and sentences are extracted.
        
        Args:
            text: Text to segment
            
        Returns:
            A list of lists of tokens, where each inner list represents a tokenized sentence
        """
        # Split text into passages (non-empty lines)
        passages = self._split_text_into_passages(text)
        
        # Process each passage and extract sentences
        all_sentences = []
        min_sentence_length = self.filters.get('min_sentence_length', 1)
        
        for doc in tqdm(self.nlp.pipe(passages, batch_size=self.batch_size), 
                        total=len(passages)):
            for sent in doc.sents:
                if sent.text.strip():
                    tokens = [token.text for token in self._filter_tokens(sent)]
                    if len(tokens) >= min_sentence_length:
                        all_sentences.append(tokens)
                    
        return all_sentences


class JiebaSegmenter(SegmentationWrapper):
    """Segmentation wrapper for Jieba Chinese text segmentation."""
    
    def __init__(self, 
                 user_dict_path: str = None,
                 pos_tagging: bool = False,
                 filters: Dict[str, Any] = None):
        """Initialize the Jieba segmenter.
        
        Args:
            user_dict_path: Path to a user dictionary file for Jieba
            pos_tagging: Whether to include POS tagging in segmentation
            filters: Dictionary of filters to apply during segmentation
                - min_sentence_length: Minimum length of sentences to include (default 1)
                - min_token_length: Minimum length of tokens to include (default 1)
                - excluded_pos: List of POS tags to exclude (if pos_tagging is True)
                - stopwords: List of stopwords to exclude
        """
        super().__init__(filters)
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
        min_length = self.filters.get('min_token_length', 1)
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
    
    def segment(self, text: Union[str, List[str]]) -> Union[List[str], List[List[str]]]:
        """Segment text(s) into tokens, removing unwanted tokens.
        
        Args:
            text: Single text or list of texts to segment
            
        Returns:
            A list of tokens for a single text, or a list of lists of tokens for multiple texts
        """
        # Handle single text case
        if isinstance(text, str):
            if self.pos_tagging:
                # With POS tagging
                tokens = list(self.pseg.cut(text))
                return self._filter_tokens(tokens)
            else:
                # Without POS tagging
                tokens = list(self.jieba.cut(text))
                return self._filter_tokens(tokens)
        
        # Handle multiple texts case
        results = []
        for single_text in tqdm(text, desc="Segmenting texts"):
            if self.pos_tagging:
                # With POS tagging
                tokens = list(self.pseg.cut(single_text))
                filtered_tokens = self._filter_tokens(tokens)
            else:
                # Without POS tagging
                tokens = list(self.jieba.cut(single_text))
                filtered_tokens = self._filter_tokens(tokens)
            
            results.append(filtered_tokens)
        
        return results
    
    def segment_to_sentences(self, text: str) -> List[List[str]]:
        """Segment text into sentences, where each sentence is a list of tokens.
        
        The text is first split into passages (non-empty lines), then each 
        passage is split into sentences, and each sentence is tokenized.
        
        Args:
            text: Text to segment
            
        Returns:
            A list of lists of tokens, where each inner list represents a tokenized sentence
        """
        
        # Simple Chinese sentence-ending punctuation pattern
        sentence_end_pattern = r"[。！？\.!?]+"
        
        # Split text into passages (non-empty lines)
        passages = self._split_text_into_passages(text)
        
        all_sentences = []
        min_sentence_length = self.filters.get('min_sentence_length', 1)
        
        for passage in passages:
            # Split passage into sentences
            if not passage:
                continue
                
            # Split by sentence-ending punctuation, but keep the punctuation
            sentences = re.split(f'({sentence_end_pattern})', passage)
            
            # Combine each sentence with its ending punctuation
            combined_sentences = []
            for i in range(0, len(sentences) - 1, 2):
                if i + 1 < len(sentences):
                    combined_sentences.append(sentences[i] + sentences[i + 1])
                else:
                    combined_sentences.append(sentences[i])
            
            # If the split didn't work (no punctuation found), use the whole passage
            if not combined_sentences:
                combined_sentences = [passage]
            
            # Process each sentence
            for sentence in combined_sentences:
                if not sentence.strip():
                    continue
                    
                if self.pos_tagging:
                    # With POS tagging
                    tokens = list(self.pseg.cut(sentence))
                    filtered_tokens = self._filter_tokens(tokens)
                else:
                    # Without POS tagging
                    tokens = list(self.jieba.cut(sentence))
                    filtered_tokens = self._filter_tokens(tokens)
                
                if len(filtered_tokens) >= min_sentence_length:
                    all_sentences.append(filtered_tokens)
        
        return all_sentences


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
                 device: str = None,
                 remove_special_tokens: bool = True,
                 filters: Dict[str, Any] = None):
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
            filters: Dictionary of filters to apply during segmentation
        """
        super().__init__(filters)
        self.batch_size = batch_size
        self.remove_special_tokens = remove_special_tokens
        
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
        except ImportError:
            raise ImportError("transformers and torch are not installed. "
                             "Please install them with 'pip install transformers torch'")
        
        self.torch = torch
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForTokenClassification.from_pretrained(
                    model_name, 
                    num_labels=len(self.labels)
                ).to(self.device)
                print(f"Loaded model {model_name} on {self.device}")
            except Exception as e:
                raise ImportError(f"Failed to load model {model_name}. Error: {str(e)}")
        
        self.model.eval()
        print(f"Using tagging scheme: {self.labels}")
    
    def _predict_tags_batch(self, texts: List[str]) -> List[List[str]]:
        """Predict segmentation tags for each character in a batch of texts."""
        import torch
        
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
            padding=True
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)
        
        # Process predictions back to tags for each text
        all_tags = []
        for i, (pred, length) in enumerate(zip(predictions, original_lengths)):
            # Skip special tokens like [CLS] and [SEP] if configured to do so
            if self.remove_special_tokens:
                # BERT tokenization adds [CLS] at start and [SEP] at end:
                # [CLS] char1 char2 ... charN [SEP]
                # So we only need positions 1 to length (inclusive)
                tags = [self.labels[p.item()] for p in pred[1:length+1]]  # Skip [CLS], include all characters
            else:
                # Include special tokens - but still limit to the actual content length
                tags = [self.labels[p.item()] for p in pred[:length+1]]
            
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
                    current_word += token
            
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
        
        # BMES tagging scheme (or other 4-tag scheme)
        elif len(self.labels) >= 4 and all(tag in self.labels for tag in tags):
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
                        current_word = token
            
            # Add the last word if it exists
            if current_word:
                words.append(current_word)
        
        return words
    
    def segment(self, text: Union[str, List[str]]) -> Union[List[str], List[List[str]]]:
        """Segment text(s) into tokens using the BERT model.
        
        Args:
            text: Single text or list of texts to segment
            
        Returns:
            A list of tokens for a single text, or a list of lists of tokens for multiple texts
        """
        # Handle single text case
        if isinstance(text, str):
            if not text.strip():
                return []
                
            # Predict tags for each character
            tags = self._predict_tags(text)
            
            # Get tokens (characters) from the input text
            tokens = list(text)
            
            # Merge tokens based on tags
            words = self._merge_tokens_by_tags(tokens, tags)
            
            # Apply filters
            min_length = self.filters.get('min_token_length', 1)
            stopwords = set(self.filters.get('stopwords', []))
            
            return [word for word in words if len(word) >= min_length and word not in stopwords]
        
        # Handle multiple texts case with proper batching
        results = []
        
        # Process empty texts first
        processed_texts = []
        text_indices = []
        
        for i, single_text in enumerate(text):
            if not single_text.strip():
                results.append([])
            else:
                processed_texts.append(single_text)
                text_indices.append(i)
        
        # Initialize results list with proper length
        results = [[] for _ in range(len(text))]
        
        # Process non-empty texts in batches
        for i in range(0, len(processed_texts), self.batch_size):
            batch_texts = processed_texts[i:i + self.batch_size]
            batch_indices = text_indices[i:i + self.batch_size]
            
            # Get tokens and tags for batch
            batch_tokens = [list(t) for t in batch_texts]
            batch_tags = self._predict_tags_batch(batch_texts)
            
            # Process each text in the batch
            for j, (tokens, tags, idx) in enumerate(zip(batch_tokens, batch_tags, batch_indices)):
                words = self._merge_tokens_by_tags(tokens, tags)
                
                # Apply filters
                min_length = self.filters.get('min_token_length', 1)
                stopwords = set(self.filters.get('stopwords', []))
                
                filtered_words = [word for word in words 
                                if len(word) >= min_length and word not in stopwords]
                
                # Store results at the original index
                results[idx] = filtered_words
        
        return results
    
    def segment_to_sentences(self, text: str) -> List[List[str]]:
        """Segment text into sentences, where each sentence is a list of tokens.
        
        Args:
            text: Text to segment
            
        Returns:
            A list of lists of tokens, where each inner list represents a tokenized sentence
        """
        # Simple Chinese sentence-ending punctuation pattern
        sentence_end_pattern = r"[。！？\.!?]+"
        
        # Split text into passages (non-empty lines)
        passages = self._split_text_into_passages(text)
        
        all_sentences = []
        min_sentence_length = self.filters.get('min_sentence_length', 1)
        
        # Process passages in batches
        for i in range(0, len(passages), self.batch_size):
            batch_passages = passages[i:i + self.batch_size]
            
            # Extract sentences from each passage
            all_batch_sentences = []
            for passage in batch_passages:
                if not passage:
                    continue
                    
                # Split by sentence-ending punctuation, but keep the punctuation
                sentences = re.split(f'({sentence_end_pattern})', passage)
                
                # Combine each sentence with its ending punctuation
                combined_sentences = []
                for i in range(0, len(sentences) - 1, 2):
                    if i + 1 < len(sentences):
                        combined_sentences.append(sentences[i] + sentences[i + 1])
                    else:
                        combined_sentences.append(sentences[i])
                
                # If the split didn't work (no punctuation found), use the whole passage
                if not combined_sentences:
                    combined_sentences = [passage]
                
                # Keep only non-empty sentences
                combined_sentences = [s for s in combined_sentences if s.strip()]
                all_batch_sentences.extend(combined_sentences)
            
            # Skip empty batches
            if not all_batch_sentences:
                continue
                
            # Get tokens for all sentences in the batch
            batch_tags = self._predict_tags_batch(all_batch_sentences)
            
            # Process each sentence
            for tags, sentence in zip(batch_tags, all_batch_sentences):
                if not sentence.strip():
                    continue

                tokens = list(sentence)
                assert len(tags) == len(tokens)
                words = self._merge_tokens_by_tags(tokens, tags)
                
                # Apply filters
                min_length = self.filters.get('min_token_length', 1)
                stopwords = set(self.filters.get('stopwords', []))
                
                filtered_words = [word for word in words 
                               if len(word) >= min_length and word not in stopwords]
                
                if len(filtered_words) >= min_sentence_length:
                    all_sentences.append(filtered_words)
        
        return all_sentences


# Factory function to create appropriate segmenter based on the backend
def create_segmenter(backend: str = "spacy", **kwargs) -> SegmentationWrapper:
    """Create a segmenter based on the specified backend.
    
    Args:
        backend: The segmentation backend to use ('spacy', 'jieba', 'bert', etc.)
        **kwargs: Additional arguments to pass to the segmenter constructor
        
    Returns:
        An instance of a SegmentationWrapper subclass
        
    Raises:
        ValueError: If the specified backend is not supported
    """
    if backend.lower() == "spacy":
        return SpacySegmenter(**kwargs)
    elif backend.lower() == "jieba":
        return JiebaSegmenter(**kwargs)
    elif backend.lower() == "bert":
        return BertSegmenter(**kwargs)
    else:
        raise ValueError(f"Unsupported segmentation backend: {backend}")