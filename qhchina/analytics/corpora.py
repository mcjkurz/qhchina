import logging
from collections import Counter
import numpy as np
from scipy.stats import fisher_exact, chi2_contingency
from typing import List, Dict, Tuple, Union, Any
from tqdm.auto import tqdm
from ..utils import validate_filters

logger = logging.getLogger("qhchina.analytics.corpora")

def compare_corpora(corpusA: Union[List[str], List[List[str]]], 
                    corpusB: Union[List[str], List[List[str]]], 
                    method: str = 'fisher', 
                    filters: Dict = None,
                    as_dataframe: bool = True) -> List[Dict]:
    """
    Compare two corpora to identify statistically significant differences in word usage.
    
    Parameters:
      corpusA: Either a flat list of tokens or a list of sentences (each sentence being a list of tokens)
      corpusB: Either a flat list of tokens or a list of sentences (each sentence being a list of tokens)
      method (str): 'fisher' for Fisher's exact test or 'chi2' or 'chi2_corrected' for the chi-square test.
                    All tests use two-sided alternatives.
      filters (dict, optional): Dictionary of filters to apply to results:
          - 'min_count': int or tuple - Minimum count threshold(s) for a word to be included 
            (can be a single int for both corpora or tuple (min_countA, min_countB)).
            Default is 0, which includes words that appear in either corpus, even if absent in one.
          - 'max_p': float - Maximum p-value threshold for statistical significance
          - 'stopwords': list - Words to exclude from results
          - 'min_word_length': int - Minimum character length for words
      as_dataframe (bool): Whether to return a pandas DataFrame.
      
    Returns:
      If as_dataframe is True:
        pandas.DataFrame: A DataFrame containing information about each word's frequency in both corpora,
                          the p-value, and the ratio of relative frequencies.
      If as_dataframe is False:
        List[dict]: Each dict contains information about a word's frequency in both corpora,
                    the p-value, and the ratio of relative frequencies.
    
    Notes:
      Two-sided tests are used because we want to detect whether words are overrepresented in either corpus.
    """
    # Validate filter keys
    valid_filter_keys = {'min_count', 'max_p', 'stopwords', 'min_word_length'}
    validate_filters(filters, valid_filter_keys, context='compare_corpora')
    
    # Validate and print filters
    if filters:
        
        # Validate filter values
        if 'min_count' in filters:
            min_count_val = filters['min_count']
            if isinstance(min_count_val, int):
                if min_count_val < 0:
                    raise ValueError("min_count must be non-negative")
            elif isinstance(min_count_val, tuple):
                if len(min_count_val) != 2 or any(v < 0 for v in min_count_val):
                    raise ValueError("min_count tuple must have 2 non-negative values")
            else:
                raise ValueError("min_count must be an int or tuple of 2 ints")
        
        if 'max_p' in filters:
            if not isinstance(filters['max_p'], (int, float)) or filters['max_p'] < 0 or filters['max_p'] > 1:
                raise ValueError("max_p must be a number between 0 and 1")
        
        if 'stopwords' in filters:
            if not isinstance(filters['stopwords'], (list, set)):
                raise ValueError("stopwords must be a list or set")
        
        if 'min_word_length' in filters:
            if not isinstance(filters['min_word_length'], int) or filters['min_word_length'] < 1:
                raise ValueError("min_word_length must be a positive integer")
        
        # Print filters
        filter_strs = []
        if 'min_count' in filters:
            filter_strs.append(f"min_count={filters['min_count']}")
        if 'max_p' in filters:
            filter_strs.append(f"max_p={filters['max_p']}")
        if 'stopwords' in filters:
            filter_strs.append(f"stopwords=<{len(filters['stopwords'])} words>")
        if 'min_word_length' in filters:
            filter_strs.append(f"min_word_length={filters['min_word_length']}")
        logger.info(f"Filters: {', '.join(filter_strs)}")
    
    # Helper function to flatten list of sentences if needed
    def flatten(corpus):
        if not corpus:
            return []
        if isinstance(corpus[0], list): # if a list of sentences
            # Filter out empty sentences
            return [word for sentence in corpus if sentence for word in sentence]
        return corpus
    
    # Flatten corpora if they are lists of sentences
    corpusA = flatten(corpusA)
    abs_freqA = Counter(corpusA)
    totalA = sum(abs_freqA.values())
    del corpusA
    
    corpusB = flatten(corpusB)
    abs_freqB = Counter(corpusB)
    totalB = sum(abs_freqB.values())
    del corpusB
    
    if totalA == 0:
        raise ValueError("corpusA is empty or contains only empty sentences")
    if totalB == 0:
        raise ValueError("corpusB is empty or contains only empty sentences")
    
    # Create a union of all words
    all_words = set(abs_freqA.keys()).union(abs_freqB.keys())
    results = []
    
    # Get min_count from filters if available, default to 0
    min_count = filters.get('min_count', 0) if filters else 0
    if isinstance(min_count, int):
        min_count = (min_count, min_count)
    
    table = np.zeros((2, 2), dtype=np.int64)
    for word in tqdm(all_words):
        a = abs_freqA.get(word, 0)  # Count in Corpus A
        b = abs_freqB.get(word, 0)  # Count in Corpus B
        
        # Check minimum counts
        if a < min_count[0] or b < min_count[1]:
            continue
            
        c = totalA - a          # Other words in Corpus A
        d = totalB - b          # Other words in Corpus B
        
        table[:] = [[a, b], [c, d]]

        # Compute the p-value using the selected statistical test.
        if method == 'fisher':
            p_value = fisher_exact(table, alternative='two-sided')[1]
        elif method == 'chi2':
            _, p_value, _, _ = chi2_contingency(table, correction=False)
        elif method == 'chi2_corrected':
            _, p_value, _, _ = chi2_contingency(table, correction=True)
        else:
            raise ValueError("Invalid method specified. Use 'fisher' or 'chi2'")
        
        # Calculate the relative frequency ratio (avoiding division by zero)
        rel_freqA = a / totalA if totalA > 0 else 0
        rel_freqB = b / totalB if totalB > 0 else 0
        ratio = (rel_freqA / rel_freqB) if rel_freqB > 0 else np.inf
        
        results.append({
            "word": word,
            "abs_freqA": a,
            "abs_freqB": b,
            "rel_freqA": rel_freqA,
            "rel_freqB": rel_freqB,
            "rel_ratio": ratio,
            "p_value": p_value,
        })
    
    # Apply other filters if specified
    if filters:
        # Filter by p-value threshold
        if 'max_p' in filters:
            results = [result for result in results if result["p_value"] <= filters['max_p']]
        
        # Filter out stopwords
        if 'stopwords' in filters:
            results = [result for result in results if result["word"] not in filters['stopwords']]
        
        # Filter by minimum length
        if 'min_word_length' in filters:
            results = [result for result in results if len(result["word"]) >= filters['min_word_length']]
            
    if as_dataframe:
        import pandas as pd
        results = pd.DataFrame(results)
    return results