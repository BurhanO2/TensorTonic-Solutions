import numpy as np

def gini_impurity(y_left: list, y_right: list) -> float:
    """
    Returns the impurity as a float.
    """
    y_left = np.asarray(y_left)
    y_right = np.asarray(y_right)

    def n_impurity(l):
        if l.size == 0:
            return 0.0
        c = np.unique(l, return_counts=True)[1]
        p = c / l.size
        return float(1.0 - np.sum(p ** 2))
    
    total = y_left.size + y_right.size
    if total == 0:
        return 0.0
    gi = (y_left.size * n_impurity(y_left) + y_right.size * n_impurity(y_right)) / total
    return float(gi)