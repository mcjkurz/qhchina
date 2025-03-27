from collections import Counter
import numpy as np
from scipy.stats import fisher_exact, chi2_contingency
from typing import List, Dict, Tuple, Union

def compare_corpora(corpusA : List[str], 
                    corpusB : List[str], 
                    method : str = 'fisher', 
                    min_count : Union[int, Tuple[int, int]] = 1,
                    as_dataframe : bool = False) -> List[Dict]:
    """
    Compare two corpora to identify statistically significant differences in word usage.
    
    Parameters:
      corpusA (list of str): List of tokens from corpus A.
      corpusB (list of str): List of tokens from corpus B.
      method (str): 'fisher' for Fisher's exact test or 'chi2' for the chi-square test.
      min_count (int or tuple): minimum count for a word to be included in the analysis.
      as_dataframe (bool): Whether to return a pandas DataFrame.
      
    Returns:
      List[dict]: Each dict contains information about a word's frequency in both corpora,
                  the p-value, and the ratio of relative frequencies.
    """
    # Count word frequencies in each corpus
    abs_freqA = Counter(corpusA)
    abs_freqB = Counter(corpusB)
    totalA = sum(abs_freqA.values())
    totalB = sum(abs_freqB.values())
    
    # Create a union of all words
    all_words = set(abs_freqA.keys()).union(abs_freqB.keys())
    results = []
    
    for word in all_words:
        a = abs_freqA.get(word, 0)  # Count in Corpus A
        b = abs_freqB.get(word, 0)  # Count in Corpus B
        c = totalA - a          # Other words in Corpus A
        d = totalB - b          # Other words in Corpus B
        
        if isinstance(min_count, int):
            min_count = (min_count, min_count)
        if a < min_count[0] or b < min_count[1]:
           continue
        table = np.array([[a, b], [c, d]])

        # Compute the p-value using the selected statistical test.
        if method == 'fisher':
            p_value = fisher_exact(table, alternative='two-sided')[1]
        elif method == 'chi2':
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
    if as_dataframe:
        import pandas as pd
        results = pd.DataFrame(results)
    return results