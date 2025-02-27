from collections import Counter
import numpy as np
from scipy.stats import fisher_exact, chi2_contingency

def compare_corpora(corpusA, corpusB, method='fisher', min_count=1, as_dataframe=False):
    """
    Compare two corpora to identify statistically significant differences in word usage.
    
    Parameters:
      corpusA (list of str): List of tokens from corpus A.
      corpusB (list of str): List of tokens from corpus B.
      significance (float): p-value threshold for significance.
      method (str): 'fisher' for Fisher's exact test or 'chi2' for the chi-square test.
      
    Returns:
      List[dict]: Each dict contains information about a word's frequency in both corpora,
                  the p-value, and the ratio of relative frequencies.
    """
    # Count word frequencies in each corpus
    freqA = Counter(corpusA)
    freqB = Counter(corpusB)
    totalA = sum(freqA.values())
    totalB = sum(freqB.values())
    
    # Create a union of all words
    all_words = set(freqA.keys()).union(freqB.keys())
    results = []
    
    for word in all_words:
        a = freqA.get(word, 0)  # Count in Corpus A
        b = freqB.get(word, 0)  # Count in Corpus B
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
        rel_freq_A = a / totalA if totalA > 0 else 0
        rel_freq_B = b / totalB if totalB > 0 else 0
        ratio = (rel_freq_A / rel_freq_B) if rel_freq_B > 0 else np.inf
        
        results.append({
            "word": word,
            "freqA": a,
            "freqB": b,
            "rel_freqA": rel_freq_A,
            "rel_freqB": rel_freq_B,
            "rel_ratio": ratio,
            "p_value": p_value,
        })
    if as_dataframe:
        import pandas as pd
        results = pd.DataFrame(results)
    return results