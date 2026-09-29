import numpy as np

def hinge_loss(y_true: list, y_score: list, margin: float = 1.0, reduction: str = "mean") -> float:
    """
    Returns the loss as a float.
    """
    # Write code here
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    l = np.maximum(0.0, margin - y_true * y_score)
    if reduction == "mean":
        return float(np.mean(l))
    return float(np.sum(l))