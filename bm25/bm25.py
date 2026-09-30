import math
from collections import Counter
import numpy as np

def bm25_score(query_tokens: list[str], docs: list[list[str]], k1: float = 1.2, b: float = 0.75) -> np.ndarray:
    """
    Returns a NumPy array with one score per document.
    """
    # Write code here
    n = len(docs)
    
    if n == 0:
        return np.zeros(0, dtype=float)

    lengths = np.array([len(d) for d in docs], dtype=float)
    avg_length = float(np.mean(lengths))
    freqs = [Counter(d) for d in docs]
    df = Counter()

    for d in docs:
        df.update(set(d))

    scores = np.zeros(n, dtype=float)

    for term in dict.fromkeys(query_tokens):
        frequency = df[term]
        if frequency == 0:
            continue
        idf = math.log((n - frequency + 0.5) / (frequency + 0.5) + 1.0)
        tf = np.array([counts[term] for counts in freqs], dtype=float)
        length_factor = 1.0 - b + b * lengths / avg_length
        scores += idf * tf * (k1 + 1.0) / (tf + k1 * length_factor)

    return scores