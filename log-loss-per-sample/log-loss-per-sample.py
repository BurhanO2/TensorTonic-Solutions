import math

def log_loss(y_true: list, y_pred: list, eps: float = 1e-15) -> list:
    """
    Returns a list of loss values.
    """
    losses = []
    for t, p in zip(y_true, y_pred):
        clipped = max(eps, min(1 - eps, p))
        losses.append(-(t * math.log(clipped) + ((1 - t) * math.log(1 - clipped))))
    return losses