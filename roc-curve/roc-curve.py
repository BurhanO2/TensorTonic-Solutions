import numpy as np

def roc_curve(y_true: list, y_score: list) -> dict:
    """
    Returns a dictionary with fpr, tpr, and thresholds.
    """
    # Write code here
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    order = np.argsort(-y_score, kind="stable")
    sorted_labels = y_true[order]
    sorted_scores = y_score[order]

    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    last_for_score = np.r_[np.flatnonzero(np.diff(sorted_scores)), sorted_scores.size - 1]
    positives = np.sum(y_true == 1)
    negatives = np.sum(y_true == 0)

    fpr = np.r_[0.0, fp[last_for_score] / negatives]
    tpr = np.r_[0.0, tp[last_for_score] / positives]
    thresholds = np.r_[np.inf, sorted_scores[last_for_score]]
    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds
    }