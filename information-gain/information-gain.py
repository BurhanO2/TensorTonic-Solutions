import numpy as np

def information_gain(y: list, split_mask: list) -> float:
    """
    Returns the information gain as a float.
    """
    y = np.asarray(y)
    split_mask = np.asarray(split_mask, dtype=bool)

    def entropy(x):
        if x.size == 0:
            return 0.0
        c = np.unique(x, return_counts=True)[1]
        p = c / x.size
        return float(-np.sum(p * np.log2(p)))

    l = y[split_mask]
    r = y[~split_mask]

    if l.size == 0 or r.size == 0:
        return 0.0
    w = (l.size * entropy(l) + r.size * entropy(r)) / y.size
    return float(entropy(y) - w)