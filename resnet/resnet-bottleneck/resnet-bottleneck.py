import numpy as np

def bottleneck_block(x, W1, W2, W3, Ws):
    """
    Returns the bottleneck residual-block output as a nested list.
    """
    x = np.array(x, dtype=float)
    W1 = np.array(W1, dtype=float)
    W2 = np.array(W2, dtype=float)
    Ws = np.array(Ws, dtype=float)
    if Ws is not None:
        Ws = np.array(Ws, dtype=float)
        i = x @ Ws
    else:
        i = x.copy()

    return np.maximum(0, (np.maximum(0, np.maximum(0, x @ W1) @ W2)) @ W3 + i)