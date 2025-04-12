from typing import List, Dict, Any, Union, Optional, Set
from tqdm.auto import tqdm
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
                 disabled: List[str] = ["ner", "lemmatizer", "attribute_ruler"],
                 batch_size: int = 200,
                 user_dict: Union[List[str], str] = None,
                 filters: Dict[str, Any] = None):
        """Initialize the spaCy segmenter.
        
        Args:
            model_name: Name of the spaCy model to use
            disabled: List of pipeline components to disable for better performance
            batch_size: Batch size for processing multiple texts
            user_dict: Custom user dictionary - either a list of words or path to a dictionary file
            filters: Dictionary of filters to apply during segmentation
                - min_sentence_length: Minimum length of sentences to include (default 1)
                - min_token_length: Minimum length of tokens to include (default 1)
                - excluded_pos: Set of POS tags to exclude from token outputs (default: NUM, SYM, SPACE)
        """
        super().__init__(filters)
        self.model_name = model_name
        self.disabled = disabled
        self.batch_size = batch_size
        self.user_dict = user_dict
        
        # Try to load the model, download if needed
        try:
            import spacy
        except ImportError:
            raise ImportError("spacy is not installed. Please install it with 'pip install spacy'")
        
        try:
            self.nlp = spacy.load(model_name, disable=self.disabled)
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
                self.nlp = spacy.load(model_name, disable=self.disabled)
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
    
    def _split_text_into_chunks(self, text: str, max_length: int) -> List[str]:
        """Split text into chunks of maximum length."""
        if len(text) <= max_length:
            return [text]
            
        chunks = []
        for i in range(0, len(text), max_length):
            chunks.append(text[i:i + max_length])
        
        return chunks
    
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
                chunks = self._split_text_into_chunks(text, self.nlp.max_length)
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


# Factory function to create appropriate segmenter based on the backend
def create_segmenter(backend: str = "spacy", **kwargs) -> SegmentationWrapper:
    """Create a segmenter based on the specified backend.
    
    Args:
        backend: The segmentation backend to use ('spacy', 'jieba', etc.)
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
    else:
        raise ValueError(f"Unsupported segmentation backend: {backend}")