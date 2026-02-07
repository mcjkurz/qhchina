import logging
import os
import tempfile
from typing import List, Dict, Any, Union, Optional, Set, Tuple
from tqdm.auto import tqdm
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
    'create_segmenter',
]


class SegmentationWrapper:
    """
    Base segmentation wrapper class that can be extended for different segmentation tools.
    
    Args:
        strategy: Strategy to process texts. Options: 'line', 'sentence', 'chunk', 'whole'. 
            Default is 'whole'.
        chunk_size: Size of chunks when using 'chunk' strategy.
        filters: Dictionary of filters to apply during segmentation:
            - stopwords: List or set of stopwords to exclude (converted to set internally)
            - min_word_length: Minimum length of tokens to include (default 1)
            - excluded_pos: List or set of POS tags to exclude (converted to set internally)
        user_dict: Custom user dictionary for segmentation. Can be:
            - str: Path to a dictionary file
            - List[str]: List of words
            - List[Tuple]: List of tuples like (word, freq, pos) or (word, freq)
        sentence_end_pattern: Regular expression pattern for sentence endings (default: 
            Chinese and English punctuation).
    """
    
    # Valid filter keys
    VALID_FILTER_KEYS = {'stopwords', 'min_word_length', 'excluded_pos'}
    
    def __init__(self, strategy: str = "whole", chunk_size: int = 512, filters: Dict[str, Any] = None,
                 user_dict: Union[str, List[Union[str, Tuple]], None] = None,
                 sentence_end_pattern: str = r"([。！？\.!?……]+)"):
        if strategy is None:
            raise ValueError("strategy cannot be None")
        self.strategy = strategy.strip().lower()
        if self.strategy not in ["line", "sentence", "chunk", "whole"]:
            raise ValueError(f"Invalid segmentation strategy: {strategy}. Must be one of: line, sentence, chunk, whole")
        
        self.chunk_size = chunk_size
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
    
    def _get_user_dict_as_list(self) -> Optional[List[Union[str, Tuple]]]:
        """Get the user dictionary as a list of words/tuples.
        
        Returns:
            List of words or tuples if user_dict is provided, None otherwise.
            If user_dict is a file path, reads and parses the file.
        """
        if self.user_dict is None:
            return None
        
        if isinstance(self.user_dict, list):
            return self.user_dict
        
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
    
    def _get_user_dict_path(self, default_freq: Optional[int] = None) -> Optional[str]:
        """Get the user dictionary as a file path.
        
        If user_dict is already a path, returns it directly.
        If user_dict is a list, creates a temporary file and returns its path.
        
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
        
        # If it's a list, create a temporary file
        if isinstance(self.user_dict, list):
            try:
                # Create a temporary file that won't be auto-deleted
                fd, temp_path = tempfile.mkstemp(suffix='.txt', prefix='user_dict_')
                self._temp_dict_path = temp_path
                
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    for item in self.user_dict:
                        if isinstance(item, str):
                            # Add default frequency if specified
                            if default_freq is not None:
                                f.write(f"{item} {default_freq}\n")
                            else:
                                f.write(f"{item}\n")
                        elif isinstance(item, tuple):
                            # Write tuple as space-separated: word freq pos
                            f.write(" ".join(str(x) for x in item) + "\n")
                
                logger.debug(f"Created temporary user dictionary at {temp_path}")
                return temp_path
            except Exception as e:
                logger.error(f"Failed to create temporary user dictionary: {str(e)}")
                return None
        
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
        if text is None:
            return []
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
    """
    Segmentation wrapper for spaCy models.
    
    Note: spaCy Chinese models use spacy-pkuseg, a fork of pkuseg trained on the OntoNotes
    corpus and co-trained with downstream statistical components (POS tagging, NER, parsing).
    
    Args:
        model_name: Name of the spaCy model to use.
        disable: List of pipeline components to disable for better performance; 
            For common applications, use ["ner", "lemmatizer"]. Default is None.
        batch_size: Batch size for processing multiple texts.
        user_dict: Custom user dictionary - either a list of words/tuples or path to a 
            dictionary file.
        strategy: Strategy to process texts. Options: 'line', 'sentence', 'chunk', 'whole'.
        chunk_size: Size of chunks when using 'chunk' strategy.
        filters: Dictionary of filters to apply during segmentation:
            - min_word_length: Minimum length of tokens to include (default 1)
            - excluded_pos: Set of POS tags to exclude from token outputs
            - stopwords: Set of stopwords to exclude
        sentence_end_pattern: Regular expression pattern for sentence endings.
    """
    
    def __init__(self, model_name: str = "zh_core_web_sm", 
                 disable: Optional[List[str]] = None,
                 batch_size: int = 200,
                 user_dict: Union[str, List[Union[str, Tuple]], None] = None,
                 strategy: str = "whole", 
                 chunk_size: int = 512,
                 filters: Dict[str, Any] = None,
                 sentence_end_pattern: str = r"([。！？\.!?……]+)"):
        super().__init__(strategy=strategy, chunk_size=chunk_size, filters=filters,
                         user_dict=user_dict, sentence_end_pattern=sentence_end_pattern)
        self.model_name = model_name
        self.disable = disable or []
        self.batch_size = batch_size
        
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
        user_dict: Custom user dictionary. Can be:
            - str: Path to a dictionary file (one word per line)
            - List[str]: List of words
            - List[Tuple]: List of tuples (only first element/word is used)
        pos_tagging: Whether to include POS tagging in segmentation.
        strategy: Strategy to process texts. Options: 'line', 'sentence', 'chunk', 'whole'.
        chunk_size: Size of chunks when using 'chunk' strategy.
        filters: Dictionary of filters to apply during segmentation:
            - min_word_length: Minimum length of tokens to include (default 1)
            - excluded_pos: List of POS tags to exclude (if pos_tagging is True)
            - stopwords: Set of stopwords to exclude
        sentence_end_pattern: Regular expression pattern for sentence endings.
    """
    
    def __init__(self, 
                 model_name: str = 'default',
                 user_dict: Union[str, List[Union[str, Tuple]], None] = None,
                 pos_tagging: bool = False,
                 strategy: str = "whole",
                 chunk_size: int = 512,
                 filters: Dict[str, Any] = None,
                 sentence_end_pattern: str = r"([。！？\.!?……]+)"):
        super().__init__(strategy=strategy, chunk_size=chunk_size, filters=filters,
                         user_dict=user_dict, sentence_end_pattern=sentence_end_pattern)
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
    
    def _filter_tokens(self, tokens) -> List[str]:
        """Filter tokens based on filters."""
        min_word_length = self.filters.get('min_word_length', 1)
        stopwords = set(self.filters.get('stopwords', []))
        
        # If POS tagging is enabled and we have tokens as (word, tag) tuples
        if self.pos_tagging:
            excluded_pos = set(self.filters.get('excluded_pos', []))
            return [word for word, tag in tokens 
                    if len(word) >= min_word_length 
                    and word not in stopwords
                    and tag not in excluded_pos]
        else:
            return [token for token in tokens 
                    if len(token) >= min_word_length 
                    and token not in stopwords]
    
    def _process_all_texts(self, texts: List[str]) -> List[List[str]]:
        """Process all text units with PKUSeg and return results.
        
        Args:
            texts: List of all text units to process
            
        Returns:
            List of lists of tokens, one list per text unit
        """
        results = []
        for text_to_process in tqdm(texts, desc="Segmenting with PKUSeg"):
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
        user_dict: Custom user dictionary for Jieba. Can be:
            - str: Path to a dictionary file
            - List[str]: List of words
            - List[Tuple]: List of tuples like (word, freq, pos) or (word, freq)
        pos_tagging: Whether to include POS tagging in segmentation.
        strategy: Strategy to process texts. Options: 'line', 'sentence', 'chunk', 'whole'.
        chunk_size: Size of chunks when using 'chunk' strategy.
        filters: Dictionary of filters to apply during segmentation:
            - min_word_length: Minimum length of tokens to include (default 1)
            - excluded_pos: List of POS tags to exclude (if pos_tagging is True)
            - stopwords: Set of stopwords to exclude
        sentence_end_pattern: Regular expression pattern for sentence endings.
    """
    
    def __init__(self, 
                 user_dict: Union[str, List[Union[str, Tuple]], None] = None,
                 pos_tagging: bool = False,
                 strategy: str = "whole",
                 chunk_size: int = 512,
                 filters: Dict[str, Any] = None,
                 sentence_end_pattern: str = r"([。！？\.!?……]+)"):
        super().__init__(strategy=strategy, chunk_size=chunk_size, filters=filters,
                         user_dict=user_dict, sentence_end_pattern=sentence_end_pattern)
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
    
    def _filter_tokens(self, tokens) -> List[str]:
        """Filter tokens based on filters."""
        min_word_length = self.filters.get('min_word_length', 1)
        stopwords = set(self.filters.get('stopwords', []))
        
        # If POS tagging is enabled and we have tokens as (word, flag) tuples
        if self.pos_tagging:
            excluded_pos = set(self.filters.get('excluded_pos', []))
            return [word for word, flag in tokens 
                    if len(word) >= min_word_length 
                    and word not in stopwords
                    and flag not in excluded_pos]
        else:
            return [token for token in tokens 
                    if len(token) >= min_word_length 
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
        user_dict: Custom user dictionary (not supported for BERT segmenter, will be ignored
            with a warning).
        strategy: Strategy to process texts. Options: 'line', 'sentence', 'chunk', 'whole'.
        chunk_size: Size of chunks when using 'chunk' strategy.
        filters: Dictionary of filters to apply during segmentation:
            - min_word_length: Minimum length of tokens to include (default 1)
            - excluded_pos: Set of POS tags to exclude from token outputs
            - stopwords: Set of stopwords to exclude
        sentence_end_pattern: Regular expression pattern for sentence endings.
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
                 tagging_scheme: Union[str, List[str]] = "be",
                 batch_size: int = 32,
                 device: Optional[str] = None,
                 remove_special_tokens: bool = True,
                 max_sequence_length: int = 512,
                 user_dict: Union[str, List[Union[str, Tuple]], None] = None,
                 strategy: str = "whole",
                 chunk_size: int = 512,
                 filters: Dict[str, Any] = None,
                 sentence_end_pattern: str = r"([。！？\.!?……]+)"):
        # Use max_sequence_length as chunk_size if not provided separately
        if not chunk_size and strategy == "chunk":
            chunk_size = max_sequence_length
        
        super().__init__(strategy=strategy, chunk_size=chunk_size, filters=filters,
                         user_dict=user_dict, sentence_end_pattern=sentence_end_pattern)
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
    
    def _filter_words(self, words: List[str]) -> List[str]:
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
        user_dict: Custom user dictionary (not supported for LLM segmenter, will be ignored
            with a warning).
        strategy: Strategy to process texts. Options: 'line', 'sentence', 'chunk', 'whole'.
        chunk_size: Size of chunks when using 'chunk' strategy.
        filters: Dictionary of filters to apply during segmentation:
            - min_word_length: Minimum length of tokens to include (default 1)
            - excluded_pos: Set of POS tags to exclude from token outputs
            - stopwords: Set of stopwords to exclude
        sentence_end_pattern: Regular expression pattern for sentence endings.
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
                 user_dict: Union[str, List[Union[str, Tuple]], None] = None,
                 strategy: str = "whole",
                 chunk_size: int = 512,
                 filters: Dict[str, Any] = None,
                 sentence_end_pattern: str = r"([。！？\.!?……]+)"):
        super().__init__(strategy=strategy, chunk_size=chunk_size, filters=filters,
                         user_dict=user_dict, sentence_end_pattern=sentence_end_pattern)
        
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
    
    def _call_llm_api(self, text: str) -> List[str]:
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
    
    def _filter_tokens(self, tokens: List[str]) -> List[str]:
        """Apply filters to the tokens."""
        min_word_length = self.filters.get('min_word_length', 1)
        stopwords = set(self.filters.get('stopwords', []))
        
        return [token for token in tokens 
                if len(token) >= min_word_length 
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
        backend: The segmentation backend to use ('spacy', 'pkuseg', 'jieba', 'bert', 'llm')
        strategy: Strategy to process texts ['line', 'sentence', 'chunk', 'whole']
        chunk_size: Size of chunks when using 'chunk' strategy
        sentence_end_pattern: Regular expression pattern for sentence endings (default: Chinese and English punctuation)
        **kwargs: Additional arguments to pass to the segmenter constructor
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
    elif backend.lower() == "pkuseg":
        return PKUSegmenter(**kwargs)
    elif backend.lower() == "jieba":
        return JiebaSegmenter(**kwargs)
    elif backend.lower() == "bert":
        return BertSegmenter(**kwargs)
    elif backend.lower() == "llm":
        return LLMSegmenter(**kwargs)
    else:
        raise ValueError(f"Unsupported segmentation backend: {backend}")