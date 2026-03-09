import logging
import os
import tempfile
from typing import Any
import importlib
import importlib.util
import re
import json
import time

logger = logging.getLogger("qhchina.preprocessing.segmentation")


__all__ = [
    'SegmentationWrapper',
    'SpacySegmenter',
    'PKUSegmenter',
    'JiebaSegmenter',
    'BertSegmenter',
    'LLMSegmenter',
    'HanLPSegmenter',
    'create_segmenter',
    'print_pos_tags',
    'POS_TAGS',
]


class SegmentationWrapper:
    """
    Base segmentation wrapper class that can be extended for different segmentation tools.
    
    Args:
        strategy: Strategy to process texts. Options: 'line', 'sentence', 'chunk', 'document'. 
            Default is 'document'.
        chunk_size: Size of chunks when using 'chunk' strategy.
        chunk_overlap: Fraction of overlap between consecutive chunks (0.0 to <1.0).
            Only used when strategy is 'chunk'. Default is 0.0 (no overlap).
        filters: Dictionary of filters to apply during segmentation:
            - stopwords: List or set of stopwords to exclude (converted to set internally)
            - min_word_length: Minimum length of tokens to include (default 1)
            - excluded_pos: List or set of POS tags to exclude (converted to set internally)
        user_dict: Custom user dictionary for segmentation. Can be:
            - str: Path to a dictionary file
            - list[str]: List of words
            - list[Tuple]: List of tuples like (word, freq, pos) or (word, freq)
            - dict[str, str]: Dictionary mapping words to POS tags (e.g., {'word': 'n'})
        sentence_end_pattern: Regular expression pattern for sentence endings (default: 
            Chinese and English punctuation).
    """
    
    # Valid filter keys
    VALID_FILTER_KEYS = {'stopwords', 'min_word_length', 'excluded_pos'}
    
    def __init__(self, strategy: str = "document", chunk_size: int = 512,
                 chunk_overlap: float = 0.0,
                 filters: dict[str, Any] | None = None,
                 user_dict: str | list[str | tuple] | dict[str, str] | None = None,
                 sentence_end_pattern: str = r"([。！？\.!?……]+)"):
        if strategy is None:
            raise ValueError("strategy cannot be None")
        self.strategy = strategy.strip().lower()
        if self.strategy not in ["line", "sentence", "chunk", "document"]:
            raise ValueError(f"Invalid segmentation strategy: {strategy}. Must be one of: line, sentence, chunk, document")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        if self.strategy == "chunk" and not (0 <= self.chunk_overlap < 1):
            raise ValueError("chunk_overlap must be in [0, 1) when strategy is 'chunk'")
        self.filters = filters or {}
        self.user_dict = user_dict
        self._temp_dict_path = None  # Track temporary file for cleanup
        
        # Validate filter keys
        self._validate_filters()
        
        self.filters.setdefault('stopwords', set())
        if not isinstance(self.filters['stopwords'], set):
            self.filters['stopwords'] = set(self.filters['stopwords'])
        self.filters.setdefault('min_word_length', 1)
        self.filters.setdefault('excluded_pos', set())
        if not isinstance(self.filters['excluded_pos'], set):
            self.filters['excluded_pos'] = set(self.filters['excluded_pos'])
        self.sentence_end_pattern = sentence_end_pattern
    
    def _validate_filters(self):
        """Validate that all filter keys are recognized."""
        if not self.filters:
            return
        
        invalid_keys = set(self.filters.keys()) - self.VALID_FILTER_KEYS
        if invalid_keys:
            raise ValueError(
                f"Invalid filter key(s): {invalid_keys}. "
                f"Valid filter keys are: {self.VALID_FILTER_KEYS}"
            )
    
    def _get_user_dict_as_list(self) -> list[str | tuple] | None:
        """Get the user dictionary as a list of words/tuples.
        
        Returns:
            List of words or tuples if user_dict is provided, None otherwise.
            If user_dict is a file path, reads and parses the file.
            If user_dict is a dict, converts to list of (word, tag) tuples.
        """
        if self.user_dict is None:
            return None
        
        if isinstance(self.user_dict, list):
            return self.user_dict
        
        # user_dict is a dict - convert to list of tuples
        if isinstance(self.user_dict, dict):
            return [(word, tag) for word, tag in self.user_dict.items()]
        
        # user_dict is a file path - read and parse it
        if isinstance(self.user_dict, str):
            words = []
            try:
                with open(self.user_dict, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) == 1:
                            words.append(parts[0])
                        else:
                            # Convert to tuple: (word, freq, pos) or (word, freq)
                            words.append(tuple(parts))
                return words
            except Exception as e:
                logger.error(f"Failed to read user dictionary from file: {str(e)}")
                return None
        
        return None
    
    def _get_user_dict_path(self, default_freq: int | None = None) -> str | None:
        """Get the user dictionary as a file path.
        
        If user_dict is already a path, returns it directly.
        If user_dict is a list or dict, creates a temporary file and returns its path.
        
        Args:
            default_freq: Default frequency to use for words without frequency.
                         If None, no frequency is added (just word per line).
                         For Jieba, a reasonable value is around 100000 to ensure
                         custom words are preferred over their component parts.
        
        Returns:
            Path to the user dictionary file, or None if no user_dict is provided.
        """
        if self.user_dict is None:
            return None
        
        # If it's already a file path, return it directly
        if isinstance(self.user_dict, str):
            return self.user_dict
        
        # Convert dict or list to list format using base method
        user_dict_list = self._get_user_dict_as_list()
        if not user_dict_list:
            return None
        
        # Create a temporary file from the list
        try:
            fd, temp_path = tempfile.mkstemp(suffix='.txt', prefix='user_dict_')
            self._temp_dict_path = temp_path
            
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                for item in user_dict_list:
                    if isinstance(item, str):
                        # Add default frequency if specified
                        if default_freq is not None:
                            f.write(f"{item} {default_freq}\n")
                        else:
                            f.write(f"{item}\n")
                    elif isinstance(item, tuple):
                        # Handle (word, tag) from dict conversion - need to add freq for Jieba
                        if len(item) == 2 and default_freq is not None:
                            # From dict: (word, tag) -> write as "word freq tag"
                            word, tag = item
                            if tag:
                                f.write(f"{word} {default_freq} {tag}\n")
                            else:
                                f.write(f"{word} {default_freq}\n")
                        else:
                            # Write tuple as space-separated: word freq pos
                            f.write(" ".join(str(x) for x in item) + "\n")
            
            logger.debug(f"Created temporary user dictionary at {temp_path}")
            return temp_path
        except Exception as e:
            logger.error(f"Failed to create temporary user dictionary: {str(e)}")
            return None
    
    def _cleanup_temp_files(self):
        """Clean up any temporary files created for user dictionary."""
        temp_path = getattr(self, '_temp_dict_path', None)
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                logger.debug(f"Cleaned up temporary user dictionary at {temp_path}")
                self._temp_dict_path = None
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {temp_path}: {str(e)}")
    
    def close(self):
        """Clean up resources. Call this when done with the segmenter."""
        self._cleanup_temp_files()
    
    def reset_user_dict(self):
        """Reset the user dictionary to default state.
        
        This clears any custom words that were added via user_dict.
        Subclasses should override this method to implement backend-specific reset logic.
        """
        self._cleanup_temp_files()
        self.user_dict = None
        logger.info("User dictionary has been reset")
    
    def __del__(self):
        """Destructor to clean up temporary files."""
        self._cleanup_temp_files()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clean up resources."""
        self.close()
        return False
    
    def segment(self, text: str) -> list[str] | list[list[str]]:
        """Segment text into tokens based on the selected strategy.
        
        Args:
            text: Text to segment
            
        Returns:
            If strategy is 'document': A single list of tokens
            If strategy is 'line', 'sentence', or 'chunk': A list of lists, where each inner list
            contains tokens for a line, sentence, or chunk respectively
        """
        # Split text based on the strategy
        units = self._split_text_by_strategy(text)
        
        # Process all units
        processed_results = self._process_all_texts(units)
        
        # For 'document' strategy, merge all results into a single list
        if self.strategy == "document" and processed_results:
            return processed_results[0]
        
        return processed_results
    
    def _split_text_by_strategy(self, text: str) -> list[str]:
        """Split text based on the selected strategy.
        
        Args:
            text: The text to split
            
        Returns:
            List of text units (lines, sentences, chunks, or whole text)
        """
        if text is None:
            return []
        if self.strategy == "line":
            return self._split_into_lines(text)
        elif self.strategy == "sentence":
            return self._split_into_sentences(text)
        elif self.strategy == "chunk":
            return self._split_into_chunks(text, self.chunk_size)
        elif self.strategy == "document":
            return [text] if text.strip() else []
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}")
    
    def _split_into_lines(self, text: str) -> list[str]:
        """Split text into non-empty lines.
        
        Args:
            text: Text to split into lines
            
        Returns:
            List of non-empty lines
        """
        return [line.strip() for line in text.split('\n') if line.strip()]
    
    def _split_into_chunks(self, text: str, chunk_size: int) -> list[str]:
        """Split text into chunks of specified size with optional overlap.
        
        Delegates to :func:`qhchina.helpers.texts.split_into_chunks`.
        
        Args:
            text: Text to split into chunks
            chunk_size: Maximum size of each chunk
            
        Returns:
            List of text chunks
        """
        from qhchina.helpers.texts import split_into_chunks
        if not text:
            return []
        return split_into_chunks(text, chunk_size=chunk_size, overlap=self.chunk_overlap)
    
    def _split_into_sentences(self, text: str) -> list[str]:
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
    
    def _process_all_texts(self, texts: list[str]) -> list[list[str]]:
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
    """
    Segmentation wrapper for spaCy models.
    
    Note: spaCy Chinese models use spacy-pkuseg, a fork of pkuseg trained on the OntoNotes
    corpus and co-trained with downstream statistical components (POS tagging, NER, parsing).
    
    Args:
        model_name: Name of the spaCy model to use.
        disable: List of pipeline components to disable for better performance; 
            For common applications, use ["ner", "lemmatizer"]. Default is None.
        batch_size: Batch size for processing multiple texts.
        max_doc_length: Maximum document length before internal chunking. Documents longer
            than this will be split into chunks for processing to avoid memory issues.
            Default is 100000 characters (~100KB). Set to None to disable chunking.
        **kwargs: Base class arguments forwarded to :class:`SegmentationWrapper`
            (strategy, chunk_size, chunk_overlap, filters, user_dict, sentence_end_pattern).
    """
    
    def __init__(self, model_name: str = "zh_core_web_sm", 
                 disable: list[str] | None = None,
                 batch_size: int = 200,
                 max_doc_length: int | None = 100000,
                 **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.disable = disable or []
        self.batch_size = batch_size
        self.max_doc_length = max_doc_length
        
        # Try to load the model, download if needed
        try:
            import spacy
        except ImportError:
            raise ImportError("spacy is not installed. Please install it with 'pip install spacy'")
        
        logger.info(f"Loading spaCy model '{model_name}'... This may take a moment.")
        try:
            self.nlp = spacy.load(model_name, disable=self.disable)
            logger.info(f"Model '{model_name}' loaded successfully.")
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
                logger.info(f"Model '{model_name}' successfully downloaded and loaded.")
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
                # Get user dict as a list (handles both file paths and lists)
                words_list = self._get_user_dict_as_list()
                if words_list:
                    # Extract just the words (first element if tuple, or the string itself)
                    words = []
                    for item in words_list:
                        if isinstance(item, str):
                            words.append(item)
                        elif isinstance(item, tuple) and len(item) > 0:
                            words.append(item[0])  # First element is the word
                    
                    self.nlp.tokenizer.pkuseg_update_user_dict(words)
                    logger.info(f"Updated user dictionary with {len(words)} words")
            except Exception as e:
                logger.error(f"Failed to update user dictionary: {str(e)}")
        else:
            logger.warning("This spaCy model's tokenizer does not support pkuseg_update_user_dict")
    
    def reset_user_dict(self):
        """Reset the spaCy tokenizer's user dictionary.
        
        This clears any custom words that were added via pkuseg_update_user_dict.
        Note: This resets to an empty user dictionary, not the original state if one was loaded.
        """
        # Clean up temp files first
        self._cleanup_temp_files()
        self.user_dict = None
        
        # Reset pkuseg user dictionary if supported
        if hasattr(self.nlp.tokenizer, 'pkuseg_update_user_dict'):
            try:
                self.nlp.tokenizer.pkuseg_update_user_dict([], reset=True)
                logger.info("spaCy pkuseg user dictionary has been reset")
            except Exception as e:
                logger.error(f"Failed to reset spaCy user dictionary: {str(e)}")
        else:
            logger.warning("This spaCy model's tokenizer does not support pkuseg_update_user_dict")
    
    def _filter_tokens(self, tokens):
        """Filter tokens based on excluded POS tags and minimum length."""
        min_word_length = self.filters.get('min_word_length', 1)
        excluded_pos = self.filters.get('excluded_pos', set())
        if not isinstance(excluded_pos, set):
            excluded_pos = set(excluded_pos)
        stopwords = self.filters.get('stopwords', set())
        if not isinstance(stopwords, set):
            stopwords = set(stopwords)
        return [token for token in tokens 
                if token.pos_ not in excluded_pos 
                and len(token.text) >= min_word_length
                and token.text not in stopwords
                and token.text.strip()]
    
    def _chunk_large_text(self, text: str) -> list[str]:
        """Split a large text into smaller chunks for memory-efficient processing.
        
        Tries to split at sentence boundaries when possible to avoid breaking words.
        
        Args:
            text: The text to chunk
            
        Returns:
            List of text chunks
        """
        if self.max_doc_length is None or len(text) <= self.max_doc_length:
            return [text]
        
        chunks = []
        remaining = text
        
        while remaining:
            if len(remaining) <= self.max_doc_length:
                chunks.append(remaining)
                break
            
            # Find a good split point (sentence boundary) near max_doc_length
            chunk = remaining[:self.max_doc_length]
            
            # Look for sentence-ending punctuation to split at
            split_pos = self.max_doc_length
            for punct in ['。', '！', '？', '!', '?', '\n']:
                last_punct = chunk.rfind(punct)
                if last_punct > self.max_doc_length // 2:
                    split_pos = last_punct + 1
                    break
            
            chunks.append(remaining[:split_pos])
            remaining = remaining[split_pos:]
        
        if len(chunks) > 1:
            logger.debug(f"Split large text ({len(text)} chars) into {len(chunks)} chunks")
        
        return chunks
    
    def _process_all_texts(self, texts: list[str]) -> list[list[str]]:
        """Process all texts with spaCy's pipe and return results.
        
        Large texts are automatically chunked to avoid memory issues.
        
        Args:
            texts: List of all text units to process
            
        Returns:
            List of lists of tokens, one list per text unit
        """
        results = []
        
        for text in texts:
            # Chunk large texts to avoid memory issues
            chunks = self._chunk_large_text(text)
            
            if len(chunks) == 1:
                # Single chunk - process normally
                doc = self.nlp(chunks[0])
                filtered_tokens = [token.text for token in self._filter_tokens(doc)]
                results.append(filtered_tokens)
            else:
                # Multiple chunks - process each and combine results
                all_tokens = []
                for doc in self.nlp.pipe(chunks, batch_size=self.batch_size):
                    filtered_tokens = [token.text for token in self._filter_tokens(doc)]
                    all_tokens.extend(filtered_tokens)
                results.append(all_tokens)
        
        return results


class PKUSegmenter(SegmentationWrapper):
    """
    Segmentation wrapper for PKUSeg Chinese text segmentation.
    
    PKUSeg is a toolkit for multi-domain Chinese word segmentation developed by
    Peking University. It uses the original pkuseg package with its own pre-trained
    models (different from spacy-pkuseg, which is trained on OntoNotes).
    
    Note: PKUSeg does not support dynamic user dictionary updates. The user dictionary
    is loaded at initialization time. To change the dictionary, call reset_user_dict()
    which will reinitialize the segmenter.
    
    Args:
        model_name: Name of the model to use. Options:
            - 'default': General domain model (default)
            - 'news': News domain
            - 'web': Web domain  
            - 'medicine': Medical domain
            - 'tourism': Tourism domain
            - Or a path to a custom model directory
        pos_tagging: Whether to include POS tagging in segmentation.
        **kwargs: Base class arguments forwarded to :class:`SegmentationWrapper`
            (strategy, chunk_size, chunk_overlap, filters, user_dict, sentence_end_pattern).
    """
    
    def __init__(self, 
                 model_name: str = 'default',
                 pos_tagging: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.pos_tagging = pos_tagging
        
        # Try to import pkuseg
        try:
            import pkuseg
        except ImportError:
            raise ImportError("pkuseg is not installed. Please install it with 'pip install pkuseg'")
        
        self.pkuseg_module = pkuseg
        
        # Initialize the segmenter
        self._init_segmenter()
    
    def _init_segmenter(self):
        """Initialize or reinitialize the PKUSeg segmenter."""
        # Clean up any existing temp files before creating new ones
        self._cleanup_temp_files()
        
        # Get user dict path if provided (this may create a temp file)
        dict_path = self._get_user_dict_path() if self.user_dict else None
        
        # Build kwargs for pkuseg initialization
        kwargs = {
            'postag': self.pos_tagging
        }
        
        # Add model_name (pkuseg uses 'default' internally for None)
        if self.model_name and self.model_name != 'default':
            kwargs['model_name'] = self.model_name
        
        # Add user_dict if provided
        if dict_path:
            kwargs['user_dict'] = dict_path
            logger.info(f"Loading user dictionary from {dict_path}")
        
        logger.info(f"Initializing PKUSeg with model='{self.model_name}', postag={self.pos_tagging}")
        try:
            self.seg = self.pkuseg_module.pkuseg(**kwargs)
            logger.info("PKUSeg initialized successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize PKUSeg: {str(e)}")
    
    def reset_user_dict(self):
        """Reset the user dictionary by reinitializing PKUSeg without a user dict.
        
        Note: PKUSeg doesn't support dynamic dictionary updates, so we reinitialize
        the entire segmenter. This is different from Jieba where we can reset the
        global state.
        """
        # Clean up temp files
        self._cleanup_temp_files()
        
        # Clear user_dict reference
        self.user_dict = None
        
        # Reinitialize the segmenter without user dict
        self._init_segmenter()
        logger.info("PKUSeg user dictionary has been reset")
    
    def _get_user_dict_path(self, default_freq: int | None = None) -> str | None:
        """Get the user dictionary as a file path for PKUSeg.
        
        PKUSeg requires a simple format: one word per line, no frequency or tags.
        This overrides the base class method to produce PKUSeg-compatible output.
        
        Args:
            default_freq: Ignored for PKUSeg (included for API compatibility).
        
        Returns:
            Path to the user dictionary file, or None if no user_dict is provided.
        """
        if self.user_dict is None:
            return None
        
        # If it's already a file path, return it directly
        if isinstance(self.user_dict, str):
            return self.user_dict
        
        # Convert dict or list to list format using base method
        user_dict_list = self._get_user_dict_as_list()
        if not user_dict_list:
            return None
        
        # Create a temporary file with word-only format (PKUSeg requirement)
        try:
            fd, temp_path = tempfile.mkstemp(suffix='.txt', prefix='user_dict_')
            self._temp_dict_path = temp_path
            
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                for item in user_dict_list:
                    if isinstance(item, tuple):
                        # Extract just the word, ignore freq/tag
                        word = item[0]
                    else:
                        word = str(item)
                    f.write(f"{word}\n")
            
            logger.debug(f"Created PKUSeg user dictionary at {temp_path}")
            return temp_path
        except Exception as e:
            logger.error(f"Failed to create temporary user dictionary: {str(e)}")
            return None
    
    def _filter_tokens(self, tokens) -> list[str]:
        """Filter tokens based on filters."""
        min_word_length = self.filters.get('min_word_length', 1)
        stopwords = set(self.filters.get('stopwords', []))
        
        # If POS tagging is enabled and we have tokens as (word, tag) tuples
        if self.pos_tagging:
            excluded_pos = set(self.filters.get('excluded_pos', []))
            return [word for word, tag in tokens 
                    if len(word) >= min_word_length 
                    and word not in stopwords
                    and tag not in excluded_pos
                    and word.strip()]
        else:
            return [token for token in tokens 
                    if len(token) >= min_word_length 
                    and token not in stopwords
                    and token.strip()]
    
    def _process_all_texts(self, texts: list[str]) -> list[list[str]]:
        """Process all text units with PKUSeg and return results.
        
        Args:
            texts: List of all text units to process
            
        Returns:
            List of lists of tokens, one list per text unit
        """
        results = []
        for text_to_process in texts:
            # Skip empty text
            if not text_to_process.strip():
                results.append([])
                continue
            
            # Segment the text
            tokens = self.seg.cut(text_to_process)
            
            # Filter tokens
            filtered_tokens = self._filter_tokens(tokens)
            
            # Add to results
            results.append(filtered_tokens)
        
        return results


class JiebaSegmenter(SegmentationWrapper):
    """
    Segmentation wrapper for Jieba Chinese text segmentation.
    
    Args:
        pos_tagging: Whether to include POS tagging in segmentation.
        **kwargs: Base class arguments forwarded to :class:`SegmentationWrapper`
            (strategy, chunk_size, chunk_overlap, filters, user_dict, sentence_end_pattern).
    """
    
    def __init__(self, 
                 pos_tagging: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
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
        if self.user_dict is not None:
            self._load_user_dict()
    
    def _load_user_dict(self):
        """Load user dictionary into Jieba."""
        try:
            # Get user dict as file path (creates temp file if needed)
            # Use default_freq=100000 for words without explicit frequency
            # This ensures custom words are preferred over their component parts
            dict_path = self._get_user_dict_path(default_freq=100000)
            if dict_path:
                self.jieba.load_userdict(dict_path)
                logger.info(f"Loaded user dictionary from {dict_path}")
        except Exception as e:
            logger.error(f"Failed to load user dictionary: {str(e)}")
    
    def reset_user_dict(self):
        """Reset Jieba's dictionary to default state.
        
        This reinitializes Jieba, clearing any custom words that were added.
        Note: Jieba uses a global state, so this affects all JiebaSegmenter instances.
        """
        # Clean up temp files first
        self._cleanup_temp_files()
        self.user_dict = None
        
        # Reset Jieba to default dictionary
        try:
            # Clear user word tag table
            if hasattr(self.jieba, 'user_word_tag_tab'):
                self.jieba.user_word_tag_tab.clear()
            
            # Remove the default cache file to force rebuild from default dictionary
            # Jieba typically caches at tempdir/jieba.cache
            default_cache = os.path.join(tempfile.gettempdir(), 'jieba.cache')
            if os.path.exists(default_cache):
                try:
                    os.unlink(default_cache)
                    logger.debug(f"Removed Jieba cache file: {default_cache}")
                except Exception as e:
                    logger.warning(f"Could not remove cache file: {e}")
            
            # Also check for any cache file in the current tokenizer
            if hasattr(self.jieba, 'dt') and hasattr(self.jieba.dt, 'cache_file'):
                cache_file = self.jieba.dt.cache_file
                if cache_file and cache_file != default_cache and os.path.exists(cache_file):
                    try:
                        os.unlink(cache_file)
                        logger.debug(f"Removed Jieba cache file: {cache_file}")
                    except Exception as e:
                        logger.warning(f"Could not remove cache file: {e}")
            
            # Create a fresh tokenizer to reset the frequency dictionary
            self.jieba.dt = self.jieba.Tokenizer()
            self.jieba.dt.initialize()
            
            # Reassign module-level function references to point to the new tokenizer
            # (jieba.cut, jieba.cut_for_search, etc. are bound methods that still
            # reference the old tokenizer after we replace jieba.dt)
            self.jieba.cut = self.jieba.dt.cut
            self.jieba.cut_for_search = self.jieba.dt.cut_for_search
            self.jieba.tokenize = self.jieba.dt.tokenize
            
            # Also reassign dictionary management methods
            self.jieba.load_userdict = self.jieba.dt.load_userdict
            self.jieba.add_word = self.jieba.dt.add_word
            self.jieba.del_word = self.jieba.dt.del_word
            self.jieba.suggest_freq = self.jieba.dt.suggest_freq
            
            # Also update posseg's tokenizer reference (pseg.dt.tokenizer points to jieba.dt)
            self.pseg.dt.tokenizer = self.jieba.dt
            
            logger.info("Jieba dictionary has been reset to default state")
        except Exception as e:
            logger.error(f"Failed to reset Jieba dictionary: {str(e)}")
    
    def _filter_tokens(self, tokens) -> list[str]:
        """Filter tokens based on filters."""
        min_word_length = self.filters.get('min_word_length', 1)
        stopwords = set(self.filters.get('stopwords', []))
        
        # If POS tagging is enabled and we have tokens as (word, flag) tuples
        if self.pos_tagging:
            excluded_pos = set(self.filters.get('excluded_pos', []))
            return [word for word, flag in tokens 
                    if len(word) >= min_word_length 
                    and word not in stopwords
                    and flag not in excluded_pos
                    and word.strip()]
        else:
            return [token for token in tokens 
                    if len(token) >= min_word_length 
                    and token not in stopwords
                    and token.strip()]
    
    def _process_all_texts(self, texts: list[str]) -> list[list[str]]:
        """Process all text units with Jieba and return results.
        
        Args:
            texts: List of all text units to process
            
        Returns:
            List of lists of tokens, one list per text unit
        """
        results = []
        for text_to_process in texts:
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
    """
    Segmentation wrapper for BERT-based Chinese word segmentation.
    
    Args:
        model_name: Name of the pre-trained BERT model to load (optional if model and 
            tokenizer are provided).
        model: Pre-initialized model instance (optional if model_name is provided).
        tokenizer: Pre-initialized tokenizer instance (optional if model_name is provided).
        tagging_scheme: Either a string ('be', 'bmes') or a list of tags in their exact 
            order (e.g. ["B", "E"]). When a list is provided, the order of tags matters 
            as it maps to prediction indices.
        batch_size: Batch size for processing.
        device: Device to use ('cpu', 'cuda', etc.).
        remove_special_tokens: Whether to remove special tokens (CLS, SEP) from output. 
            Default is True, which works for BERT-based models.
        max_sequence_length: Maximum sequence length for BERT models (default 512). If 
            the text is longer than this, it will be split into chunks.
        **kwargs: Base class arguments forwarded to :class:`SegmentationWrapper`
            (strategy, chunk_size, chunk_overlap, filters, user_dict, sentence_end_pattern).
    """
    
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
                 tagging_scheme: str | list[str] = "be",
                 batch_size: int = 32,
                 device: str | None = None,
                 remove_special_tokens: bool = True,
                 max_sequence_length: int = 512,
                 **kwargs):
        strategy = kwargs.get('strategy', 'document')
        if not kwargs.get('chunk_size') and strategy == "chunk":
            kwargs['chunk_size'] = max_sequence_length
        
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.remove_special_tokens = remove_special_tokens
        self.max_sequence_length = max_sequence_length
        
        # Warn if user_dict is provided (not supported for BERT)
        if self.user_dict is not None:
            logger.warning("user_dict is not supported for BertSegmenter and will be ignored")
        
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
            logger.info(f"Loading provided model to {self.device}... This may take a moment.")
            self.model = model.to(self.device)
            self.tokenizer = tokenizer
            logger.info(f"Model loaded successfully on {self.device}.")
        else:
            # Load model and tokenizer from pretrained
            logger.info(f"Loading BERT model '{model_name}'... This may take a moment.")
            try:
                self.tokenizer = self.AutoTokenizer.from_pretrained(model_name)
                self.model = self.AutoModelForTokenClassification.from_pretrained(
                    model_name, 
                    num_labels=len(self.labels)
                ).to(self.device)
                logger.info(f"Model '{model_name}' loaded successfully on {self.device}.")
            except Exception as e:
                raise ImportError(f"Failed to load model {model_name}. Error: {str(e)}")
        
        self.model.eval()
        logger.info(f"Using tagging scheme: {self.labels}")
    
    def _filter_words(self, words: list[str]) -> list[str]:
        """Filter words based on specified filters.
        
        Args:
            words: List of words to filter
            
        Returns:
            Filtered list of words
        """
        min_word_length = self.filters.get('min_word_length', 1)
        stopwords = set(self.filters.get('stopwords', []))
        
        return [word for word in words 
                if len(word) >= min_word_length 
                and word not in stopwords
                and word.strip()]
    
    def _predict_tags_batch(self, texts: list[str]) -> list[list[str]]:
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
    
    def _predict_tags(self, text: str) -> list[str]:
        """Predict segmentation tags for each character in a single text."""
        return self._predict_tags_batch([text])[0]
    
    def _merge_tokens_by_tags(self, tokens: list[str], tags: list[str]) -> list[str]:
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
    
    def _process_all_texts(self, texts: list[str]) -> list[list[str]]:
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
                    logger.warning(f"Tags and tokens length mismatch. Tags: {len(tags)}, Tokens: {len(tokens)}")
                    results.append([])  # Add empty list for this entry
                    continue
                    
                words = self._merge_tokens_by_tags(tokens, tags)
                
                # Apply filters
                filtered_words = self._filter_words(words)
                
                # Add to results
                results.append(filtered_words)
        
        return results


class LLMSegmenter(SegmentationWrapper):
    """
    Segmentation wrapper using Language Model APIs like OpenAI.
    
    Args:
        api_key: API key for the language model service.
        model: Model name to use.
        endpoint: API endpoint URL.
        prompt: Custom prompt template with {text} placeholder (if None, uses DEFAULT_PROMPT).
        system_message: Optional system message to prepend to API calls.
        temperature: Temperature for model sampling (lower for more deterministic output).
        max_tokens: Maximum tokens in the response.
        retry_patience: Number of retries for API calls (default 1, meaning 1 retry = 
            2 total attempts).
        timeout: Timeout in seconds for API calls (default 60.0). Set to None for no timeout.
        **kwargs: Base class arguments forwarded to :class:`SegmentationWrapper`
            (strategy, chunk_size, chunk_overlap, filters, user_dict, sentence_end_pattern).
    """
    
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
                 retry_patience: int = 1,
                 timeout: float = 60.0,
                 **kwargs):
        super().__init__(**kwargs)
        
        # Warn if user_dict is provided (not supported for LLM)
        if self.user_dict is not None:
            logger.warning("user_dict is not supported for LLMSegmenter and will be ignored")
        
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint
        self.prompt = prompt or self.DEFAULT_PROMPT
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retry_patience = max(0, retry_patience)  # Number of retries (0 = no retries, 1 = one retry, etc.)
        self.timeout = timeout
        
        # Try to import OpenAI
        try:
            import openai
        except ImportError:
            raise ImportError("openai is not installed. Please install it with 'pip install openai'")
        
        # Configure OpenAI client with timeout
        if endpoint:
            # Custom API endpoint
            self.client = openai.OpenAI(
                api_key=api_key,
                base_url=endpoint,
                timeout=timeout
            )
        else:
            # Default OpenAI endpoint
            self.client = openai.OpenAI(api_key=api_key, timeout=timeout)
    
    def _call_llm_api(self, text: str) -> list[str]:
        """Call the LLM API with the provided text and parse the response as a list of tokens.
        
        Implements retry logic with exponential backoff.
        """
        prompt_text = self.prompt.format(text=text)
        
        for attempt in range(self.retry_patience + 1):  # +1 because retry_patience is number of retries
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
                            logger.warning(f"Response looks like a list but isn't valid JSON: {str(je)}")
                            logger.debug(f"Response text (first 100 chars): {response_text[:100]}...")
                    
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
                            logger.warning(f"No list found in JSON response: {parsed_json}")
                            # Fallback to raw tokens if no list found
                            return []
                    except json.JSONDecodeError as je:
                        # Show detailed error for debugging
                        logger.error(f"JSON Decode Error: {str(je)}")
                        logger.debug(f"Response text (first 100 chars): {response_text[:100]}...")
                        return []
                        
                except Exception as e:
                    logger.error(f"Error parsing API response: {str(e)}")
                    logger.debug(f"Response text (first 100 chars): {response_text[:100]}...")
                    return []
                    
            except Exception as e:
                is_last_attempt = (attempt == self.retry_patience)
                
                if is_last_attempt:
                    logger.error(f"Error calling LLM API (final attempt {attempt + 1}/{self.retry_patience}): {str(e)}")
                    return []
                else:
                    # Calculate exponential backoff delay: 2^attempt seconds
                    wait_time = 2 ** attempt
                    logger.warning(f"Error calling LLM API (attempt {attempt + 1}/{self.retry_patience}): {str(e)}")
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
    
    def _filter_tokens(self, tokens: list[str]) -> list[str]:
        """Apply filters to the tokens."""
        min_word_length = self.filters.get('min_word_length', 1)
        stopwords = set(self.filters.get('stopwords', []))
        
        return [token for token in tokens 
                if len(token) >= min_word_length 
                and token not in stopwords
                and token.strip()]
    
    def _process_all_texts(self, texts: list[str]) -> list[list[str]]:
        """Process all text units with LLM API and return results.
        
        Args:
            texts: List of all text units to process
            
        Returns:
            List of lists of tokens, one list per text unit
        """
        # Process each text unit one by one (no batching for API calls)
        results = []
        for text_to_process in texts:
            
            # Call the LLM API for this text unit
            tokens = self._call_llm_api(text_to_process)
            filtered_tokens = self._filter_tokens(tokens)
            
            # Add the tokens to results
            results.append(filtered_tokens)
                
        return results


class HanLPSegmenter(SegmentationWrapper):
    """
    Segmentation wrapper using HanLP 2.x neural tokenizers.
    
    HanLP provides state-of-the-art Chinese word segmentation using transformer models.
    It supports multiple pretrained models for different use cases (coarse/fine-grained,
    ancient Chinese, multilingual), and optionally POS tagging.
    
    As of March 2026, HanLP 2.x is incompatible with transformers>=5.0 (encode_plus
    API changes). Use: pip install "transformers<5.0" 
    
    Args:
        model: Tokenizer model to use. Can be:
            - HanLP enum value (e.g., ``hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH``)
            - String shorthand: 'coarse', 'fine', 'ctb9', 'ctb9_base', 'ancient', 
              'large', 'multilingual'
            - Full model name string (e.g., 'CTB9_TOK_ELECTRA_BASE')
            - Direct URL or path to a model
            - None for default (COARSE_ELECTRA_SMALL_ZH)
        pos_tagging: Whether to enable POS tagging. When enabled, tokens can be
            filtered using ``excluded_pos`` in filters. Default is False.
        pos_model: POS tagger model to use when ``pos_tagging=True``. Can be:
            - HanLP enum value (e.g., ``hanlp.pretrained.pos.CTB9_POS_ELECTRA_SMALL``)
            - String shorthand: 'ctb9', 'ctb5', 'pku', 'c863'
            - Full model name string (e.g., 'PKU_POS_ELECTRA_SMALL')
            - Direct URL or path to a model
            - None for default (CTB9_POS_ELECTRA_SMALL)
        dict_mode: How to apply user dictionary. Options:
            - 'force': High-priority dictionary that overrides model predictions
              (longest-prefix-matching on input text)
            - 'combine': Low-priority dictionary that combines with model predictions
              (longest-prefix-matching on output tokens)
            Default is 'force'.
        **kwargs: Base class arguments forwarded to :class:`SegmentationWrapper`
            (strategy, chunk_size, chunk_overlap, filters, user_dict, sentence_end_pattern).
    
    Example:
        >>> import hanlp
        >>> from qhchina.preprocessing import create_segmenter
        >>> # Using HanLP enum directly (recommended)
        >>> seg = create_segmenter("hanlp", model=hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH)
        >>> # Using string shorthand
        >>> seg = create_segmenter("hanlp", model="fine")
        >>> # With user dictionary
        >>> seg = create_segmenter("hanlp", user_dict={"自定义词": "n"}, dict_mode="force")
        >>> # With POS tagging and filtering
        >>> seg = create_segmenter("hanlp", pos_tagging=True, 
        ...                        filters={'excluded_pos': {'PU', 'DEG'}})
    
    Note:
        HanLP uses CTB (Chinese Treebank) POS tags by default. Common tags include:
        - NN: common noun
        - NR: proper noun
        - VV: verb
        - VA: predicative adjective
        - AD: adverb
        - P: preposition
        - DEG: associative 的
        - PU: punctuation
        - JJ: noun-modifier (adjective)
        - CD: cardinal number
        - M: measure word
    """
    
    # Convenience map for tokenizer shortcuts
    MODEL_MAP = {
        "coarse": "COARSE_ELECTRA_SMALL_ZH",
        "fine": "FINE_ELECTRA_SMALL_ZH",
        "ctb9": "CTB9_TOK_ELECTRA_SMALL",
        "ctb9_base": "CTB9_TOK_ELECTRA_BASE",
        "ancient": "KYOTO_EVAHAN_TOK_LZH",
        "large": "LARGE_ALBERT_BASE",
        "multilingual": "UD_TOK_MMINILMV2L12",
    }
    
    # Convenience map for POS tagger shortcuts
    POS_MODEL_MAP = {
        "ctb9": "CTB9_POS_ELECTRA_SMALL",
        "ctb9_base": "CTB9_POS_ALBERT_BASE",
        "ctb5": "CTB5_POS_RNN_FASTTEXT_ZH",
        "pku": "PKU_POS_ELECTRA_SMALL",
        "c863": "C863_POS_ELECTRA_SMALL",
    }
    
    def __init__(
        self,
        model: str | None = None,
        pos_tagging: bool = False,
        pos_model: str | None = None,
        dict_mode: str = "force",
        **kwargs
    ):
        super().__init__(**kwargs)
        
        try:
            import hanlp
            self._hanlp = hanlp
        except ImportError:
            raise ImportError(
                "hanlp is not installed. Please install it with 'pip install hanlp'"
            )
        
        if dict_mode not in ("force", "combine"):
            raise ValueError(f"dict_mode must be 'force' or 'combine', got '{dict_mode}'")
        self._dict_mode = dict_mode
        self.pos_tagging = pos_tagging
        
        # Resolve tokenizer model
        model_path = self._resolve_tok_model(model, hanlp)
        logger.info(f"Loading HanLP tokenizer model: {model_path}")
        self._tokenizer = hanlp.load(model_path)
        
        # Optionally load POS tagger
        self._pos_tagger = None
        if pos_tagging:
            pos_model_path = self._resolve_pos_model(pos_model, hanlp)
            logger.info(f"Loading HanLP POS tagger model: {pos_model_path}")
            self._pos_tagger = hanlp.load(pos_model_path)
        
        # Apply user dictionary if provided
        if self.user_dict is not None:
            self._apply_user_dict()
    
    def _resolve_tok_model(self, model: str | None, hanlp) -> str:
        """Resolve tokenizer model to a loadable path/URL."""
        if model is None:
            return hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH
        elif isinstance(model, str):
            if model.lower() in self.MODEL_MAP:
                model_name = self.MODEL_MAP[model.lower()]
                return getattr(hanlp.pretrained.tok, model_name)
            elif hasattr(hanlp.pretrained.tok, model):
                return getattr(hanlp.pretrained.tok, model)
            else:
                return model
        else:
            return model
    
    def _resolve_pos_model(self, pos_model: str | None, hanlp) -> str:
        """Resolve POS tagger model to a loadable path/URL."""
        if pos_model is None:
            return hanlp.pretrained.pos.CTB9_POS_ELECTRA_SMALL
        elif isinstance(pos_model, str):
            if pos_model.lower() in self.POS_MODEL_MAP:
                model_name = self.POS_MODEL_MAP[pos_model.lower()]
                return getattr(hanlp.pretrained.pos, model_name)
            elif hasattr(hanlp.pretrained.pos, pos_model):
                return getattr(hanlp.pretrained.pos, pos_model)
            else:
                return pos_model
        else:
            return pos_model
    
    def _apply_user_dict(self) -> None:
        """Apply user dictionary to tokenizer via dict_force or dict_combine."""
        user_dict_list = self._get_user_dict_as_list()
        if not user_dict_list:
            return
        
        default_tag = 'n'
        custom_dict = {}
        
        for entry in user_dict_list:
            if isinstance(entry, tuple):
                word = entry[0]
                if len(entry) == 2:
                    tag = entry[1] if entry[1] is not None else default_tag
                elif len(entry) >= 3:
                    tag = entry[2] if entry[2] else default_tag
                else:
                    tag = default_tag
            else:
                word = str(entry)
                tag = default_tag
            custom_dict[word] = tag
        
        if not custom_dict:
            return
        
        attr_name = "dict_force" if self._dict_mode == "force" else "dict_combine"
        if hasattr(self._tokenizer, attr_name):
            setattr(self._tokenizer, attr_name, custom_dict)
            logger.info(f"Applied {len(custom_dict)} entries to HanLP {attr_name}")
        else:
            logger.warning(
                f"This HanLP tokenizer model does not support {attr_name}. "
                "User dictionary will be ignored."
            )
    
    def _filter_tokens(self, tokens: list[str]) -> list[str]:
        """Apply filters to tokens (no POS tags)."""
        min_word_length = self.filters.get('min_word_length', 1)
        stopwords = self.filters.get('stopwords', set())
        
        return [
            token for token in tokens
            if len(token) >= min_word_length
            and token not in stopwords
            and token.strip()
        ]
    
    def _filter_tokens_with_pos(
        self, tokens: list[str], pos_tags: list[str]
    ) -> list[str]:
        """Apply filters to tokens with POS tags."""
        min_word_length = self.filters.get('min_word_length', 1)
        stopwords = set(self.filters.get('stopwords', []))
        excluded_pos = set(self.filters.get('excluded_pos', []))
        
        return [
            token for token, pos in zip(tokens, pos_tags)
            if len(token) >= min_word_length
            and token not in stopwords
            and pos not in excluded_pos
            and token.strip()
        ]
    
    def _process_all_texts(self, texts: list[str]) -> list[list[str]]:
        """Process all text units with HanLP and return results.
        
        HanLP supports batch processing natively for efficiency.
        When POS tagging is enabled, tokens are tagged and filtered accordingly.
        
        Args:
            texts: List of all text units to process
            
        Returns:
            List of lists of tokens, one list per text unit
        """
        tokenized = self._tokenizer(texts)
        
        if self.pos_tagging and self._pos_tagger is not None:
            pos_tagged = self._pos_tagger(tokenized)
            return [
                self._filter_tokens_with_pos(tokens, tags)
                for tokens, tags in zip(tokenized, pos_tagged)
            ]
        else:
            return [self._filter_tokens(tokens) for tokens in tokenized]


def create_segmenter(backend: str = "spacy", strategy: str = "document", chunk_size: int = 512,
                     chunk_overlap: float = 0.0,
                     sentence_end_pattern: str = r"([。！？\.!?……]+)", **kwargs) -> SegmentationWrapper:
    """Create a segmenter based on the specified backend.
    
    Args:
        backend: The segmentation backend to use ('spacy', 'pkuseg', 'jieba', 'bert', 'llm', 'hanlp')
        strategy: Strategy to process texts ['line', 'sentence', 'chunk', 'document']
        chunk_size: Size of chunks when using 'chunk' strategy
        chunk_overlap: Fraction of overlap between consecutive chunks (0.0 to <1.0).
            Only used when strategy is 'chunk'. Default is 0.0 (no overlap).
        sentence_end_pattern: Regular expression pattern for sentence endings
            (default: Chinese and English punctuation)
        **kwargs: Additional arguments to pass to the segmenter constructor
            - user_dict: Custom user dictionary. Can be:
                - str: Path to a dictionary file
                - list[str]: List of words
                - list[Tuple]: List of tuples like (word, freq, pos) or (word, freq)
                Note: Not supported for 'bert' and 'llm' backends (will log a warning)
            - filters: Dictionary of filters to apply during segmentation
                - min_word_length: Minimum length of tokens to include (default 1)
                - stopwords: Set of stopwords to exclude
                - excluded_pos: Set of POS tags to exclude (for backends that support POS tagging)
            - retry_patience: (LLM backend only) Number of retry attempts for API calls (default 1)
            - timeout: (LLM backend only) Timeout in seconds for API calls (default 60.0)
            - Other backend-specific arguments
        
    Returns:
        An instance of a SegmentationWrapper subclass
        
    Raises:
        ValueError: If the specified backend is not supported
    """
    kwargs['strategy'] = strategy
    kwargs['chunk_size'] = chunk_size
    kwargs['chunk_overlap'] = chunk_overlap
    kwargs['sentence_end_pattern'] = sentence_end_pattern
    
    backends = {
        "spacy": SpacySegmenter,
        "pkuseg": PKUSegmenter,
        "jieba": JiebaSegmenter,
        "bert": BertSegmenter,
        "llm": LLMSegmenter,
        "hanlp": HanLPSegmenter,
    }
    
    key = backend.lower()
    if key not in backends:
        raise ValueError(f"Unsupported segmentation backend: {backend}")
    return backends[key](**kwargs)


# POS tag documentation for each backend
POS_TAGS = {
    "spacy": {
        "description": "spaCy uses Universal Dependencies (UD) POS tags",
        "url": "https://universaldependencies.org/u/pos/",
        "tags": {
            "ADJ": "adjective",
            "ADP": "adposition (preposition/postposition)",
            "ADV": "adverb",
            "AUX": "auxiliary verb",
            "CCONJ": "coordinating conjunction",
            "DET": "determiner",
            "INTJ": "interjection",
            "NOUN": "noun",
            "NUM": "numeral",
            "PART": "particle",
            "PRON": "pronoun",
            "PROPN": "proper noun",
            "PUNCT": "punctuation",
            "SCONJ": "subordinating conjunction",
            "SYM": "symbol",
            "VERB": "verb",
            "X": "other",
        },
    },
    "jieba": {
        "description": "Jieba uses ICTCLAS/北大 POS tags (when pos_tagging=True)",
        "url": "https://github.com/fxsjy/jieba",
        "tags": {
            "n": "noun (名词)",
            "nr": "person name (人名)",
            "ns": "place name (地名)",
            "nt": "organization (机构团体)",
            "nz": "other proper noun (其他专名)",
            "v": "verb (动词)",
            "vd": "adverb-verb (副动词)",
            "vn": "noun-verb (名动词)",
            "a": "adjective (形容词)",
            "ad": "adverb-adjective (副形词)",
            "an": "noun-adjective (名形词)",
            "d": "adverb (副词)",
            "m": "numeral (数词)",
            "q": "classifier/measure word (量词)",
            "r": "pronoun (代词)",
            "p": "preposition (介词)",
            "c": "conjunction (连词)",
            "u": "auxiliary (助词)",
            "uj": "的",
            "uv": "地",
            "ud": "得",
            "ul": "了",
            "e": "interjection (叹词)",
            "y": "modal particle (语气词)",
            "o": "onomatopoeia (拟声词)",
            "w": "punctuation (标点符号)",
            "x": "non-morpheme (非语素字)",
            "eng": "English word",
        },
    },
    "pkuseg": {
        "description": "PKUSeg uses CTB-style POS tags (when pos_tagging=True)",
        "url": "https://github.com/lancopku/pkuseg-python",
        "tags": {
            "n": "noun (名词)",
            "nr": "person name (人名)",
            "ns": "place name (地名)",
            "nt": "organization (机构团体)",
            "nx": "nominal string (名词性字符串)",
            "nz": "other proper noun (其它专名)",
            "v": "verb (动词)",
            "vd": "adverb-verb (副动词)",
            "vn": "noun-verb (名动词)",
            "a": "adjective (形容词)",
            "ad": "adverb-adjective (副形词)",
            "an": "noun-adjective (名形词)",
            "d": "adverb (副词)",
            "m": "numeral (数词)",
            "q": "classifier (量词)",
            "r": "pronoun (代词)",
            "p": "preposition (介词)",
            "c": "conjunction (连词)",
            "u": "auxiliary (助词)",
            "xc": "other function word (其它虚词)",
            "w": "punctuation (标点符号)",
        },
    },
    "hanlp": {
        "description": "HanLP uses CTB (Chinese Treebank) POS tags by default",
        "url": "https://hanlp.hankcs.com/docs/annotations/pos/ctb.html",
        "tags": {
            "AD": "adverb (副词)",
            "AS": "aspect marker (了/着/过)",
            "BA": "把 in ba-construction",
            "CC": "coordinating conjunction (并列连词)",
            "CD": "cardinal number (基数词)",
            "CS": "subordinating conjunction (从属连词)",
            "DEC": "的 complementizer/nominalizer",
            "DEG": "的 associative/genitive",
            "DER": "得 resultative",
            "DEV": "地 adverbial",
            "DT": "determiner (限定词)",
            "ETC": "等/等等 in coordination",
            "FW": "foreign word",
            "IJ": "interjection (感叹词)",
            "JJ": "noun-modifier (形容词/名词修饰语)",
            "LB": "被 in long bei-construction",
            "LC": "localizer (方位词)",
            "M": "measure word (量词)",
            "MSP": "other particle (其它小品词)",
            "NN": "common noun (普通名词)",
            "NR": "proper noun (专有名词)",
            "NT": "temporal noun (时间名词)",
            "OD": "ordinal number (序数词)",
            "ON": "onomatopoeia (拟声词)",
            "P": "preposition (介词)",
            "PN": "pronoun (代词)",
            "PU": "punctuation (标点)",
            "SB": "被 in short bei-construction",
            "SP": "sentence-final particle (句末助词)",
            "VA": "predicative adjective (谓语形容词)",
            "VC": "copula 是",
            "VE": "有 as main verb",
            "VV": "other verb (其它动词)",
        },
        "models": {
            "ctb9 (default)": "Chinese Treebank 9.0 tags (shown above)",
            "pku": "PKU/北大 tags (similar to Jieba)",
            "c863": "C863 corpus tags",
        },
    },
}


def print_pos_tags(backend: str | None = None) -> None:
    """Print POS (Part-of-Speech) tag documentation for segmentation backends.
    
    This function displays the POS tags used by each backend that supports
    POS tagging, helping users understand which tags to use in the
    ``excluded_pos`` filter.
    
    Args:
        backend: Specific backend to show tags for. Options:
            - 'spacy': Universal Dependencies tags
            - 'jieba': ICTCLAS/北大 tags
            - 'pkuseg': CTB-style tags
            - 'hanlp': Chinese Treebank tags (default) or PKU tags
            - None: Show tags for all backends (default)
    
    Example:
        >>> from qhchina.preprocessing import print_pos_tags
        >>> print_pos_tags('hanlp')  # Show HanLP tags only
        >>> print_pos_tags()  # Show all backends
    """
    if backend is not None:
        backend = backend.lower()
        if backend not in POS_TAGS:
            valid = ", ".join(sorted(POS_TAGS.keys()))
            print(f"Unknown backend: '{backend}'. Valid options: {valid}")
            return
        backends_to_show = [backend]
    else:
        backends_to_show = list(POS_TAGS.keys())
    
    for i, name in enumerate(backends_to_show):
        if i > 0:
            print()
        info = POS_TAGS[name]
        print("=" * 70)
        print(f"  {name.upper()} POS Tags")
        print("=" * 70)
        print(f"Description: {info['description']}")
        print(f"Reference:   {info['url']}")
        print()
        
        # Print tags in a nice table format
        print("Tags:")
        print("-" * 50)
        max_tag_len = max(len(tag) for tag in info["tags"])
        for tag, desc in info["tags"].items():
            print(f"  {tag:<{max_tag_len}}  {desc}")
        
        # Print model variants if available (e.g., HanLP)
        if "models" in info:
            print()
            print("Model variants:")
            print("-" * 50)
            for model, model_desc in info["models"].items():
                print(f"  {model}: {model_desc}")
    
    print()
    print("=" * 70)
    print("Usage: filters={'excluded_pos': {'PU', 'AD', ...}}")
    print("=" * 70)