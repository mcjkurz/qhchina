from collections import Counter
import numpy as np
from scipy.stats import fisher_exact, chi2_contingency
from typing import List, Dict, Tuple, Union, Any
from tqdm.auto import tqdm

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
      filters (dict, optional): Dictionary of filters to apply to results:
          - 'min_count': int or tuple - Minimum count threshold(s) for a word to be included 
            (can be a single int for both corpora or tuple (min_countA, min_countB)).
            Default is 0, which includes words that appear in either corpus, even if absent in one.
          - 'max_p': float - Maximum p-value threshold for statistical significance
          - 'stopwords': list - Words to exclude from results
          - 'min_length': int - Minimum character length for words
      as_dataframe (bool): Whether to return a pandas DataFrame.
      
    Returns:
      If as_dataframe is True:
        pandas.DataFrame: A DataFrame containing information about each word's frequency in both corpora,
                          the p-value, and the ratio of relative frequencies.
      If as_dataframe is False:
        List[dict]: Each dict contains information about a word's frequency in both corpora,
                    the p-value, and the ratio of relative frequencies.
    """
    # Helper function to flatten list of sentences if needed
    def flatten(corpus):
        if not corpus:
            return []
        if isinstance(corpus[0], list): # if a list of sentences
            return [word for sentence in corpus for word in sentence]
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
    
    # Create a union of all words
    all_words = set(abs_freqA.keys()).union(abs_freqB.keys())
    results = []
    
    # Get min_count from filters if available, default to 0
    min_count = filters.get('min_count', 0) if filters else 0
    if isinstance(min_count, int):
        min_count = (min_count, min_count)
    
    for word in tqdm(all_words):
        a = abs_freqA.get(word, 0)  # Count in Corpus A
        b = abs_freqB.get(word, 0)  # Count in Corpus B
        
        # Check minimum counts
        if a < min_count[0] or b < min_count[1]:
            continue
            
        c = totalA - a          # Other words in Corpus A
        d = totalB - b          # Other words in Corpus B
        
        table = np.array([[a, b], [c, d]])

        # Compute the p-value using the selected statistical test.
        if method == 'fisher':
            p_value = fisher_exact(table, alternative='two-sided')[1]
        elif method == 'chi2':
            _, p_value, _, _ = chi2_contingency(table, correction=True)
        elif method == 'chi2_corrected':
            _, p_value, _, _ = chi2_contingency(table, correction=False)
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
        if 'min_length' in filters:
            results = [result for result in results if len(result["word"]) >= filters['min_length']]
            
    if as_dataframe:
        import pandas as pd
        results = pd.DataFrame(results)
    return results