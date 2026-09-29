import numpy as np

def bag_of_words_vector(tokens: list, vocab: list) -> np.ndarray:
    """
    Returns a NumPy array with length len(vocab).
    """
    # Write code here
    token_ix = {token: ix for ix, token in enumerate(vocab)}
    counts = np.zeros(len(vocab), dtype=int)
    for token in tokens:
        if token in token_ix:
            counts[token_ix[token]] += 1
    return counts
    