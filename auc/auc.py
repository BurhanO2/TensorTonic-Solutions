import numpy as np

def auc(fpr: list, tpr: list) -> float:
    """
    Returns the area as a float.
    """
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    w = np.diff(fpr)
    h = 0.5 * (tpr[:-1] + tpr[1:])
    return float(np.sum(w * h))