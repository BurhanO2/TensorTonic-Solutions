import numpy as np

def identity_block(x, W1, W2):
    """
    Returns the identity residual-block output as a nested list.
    """
    x = np.array(x, dtype=float)
    W1 = np.array(W1, dtype=float)
    W2 = np.array(W2, dtype=float)
    i = x.copy()
    out = np.maximum(0, x @ W1.T) @ W2.T
    result = np.maximum(0, out + i)
    return result