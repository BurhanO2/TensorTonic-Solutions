import numpy as np

def classification_metrics(y_true: list[int], y_pred: list[int], average: str = "micro", pos_label: int = 1) -> dict:
    """
    Returns a dictionary containing accuracy, precision, recall, and f1 rounded to six decimals.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))

    tp = np.array([np.sum((y_true == label) & (y_pred == label)) for label in labels], dtype=float)
    fp = np.array([np.sum((y_true != label) & (y_pred == label)) for label in labels], dtype=float)
    fn = np.array([np.sum((y_true == label) & (y_pred != label)) for label in labels], dtype=float)
    support = np.array([np.sum(y_true == label) for label in labels], dtype=float)

    precision_by_class = tp / np.maximum(tp + fp, 1.0)
    recall_by_class = tp / np.maximum(tp + fn, 1.0)
    f1_by_class = 2 * precision_by_class * recall_by_class / np.maximum(precision_by_class + recall_by_class, 1e-12)

    if average == "micro":
        total_tp = float(np.sum(tp))
        total_fp = float(np.sum(fp))
        total_fn = float(np.sum(fn))
        precision = total_tp / max(total_tp + total_fp, 1.0)
        recall = total_tp / max(total_tp + total_fn, 1.0)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    elif average == "macro":
        precision = float(np.mean(precision_by_class))
        recall = float(np.mean(recall_by_class))
        f1 = float(np.mean(f1_by_class))
    elif average == "weighted":
        weights = support / np.sum(support)
        precision = float(np.sum(weights * precision_by_class))
        recall = float(np.sum(weights * recall_by_class))
        f1 = float(np.sum(weights * f1_by_class))
    else:
        matches = np.where(labels == pos_label)[0]
        if len(matches) == 0:
            precision = recall = f1 = 0.0
        else:
            index = matches[0]
            precision = float(precision_by_class[index])
            recall = float(recall_by_class[index])
            f1 = float(f1_by_class[index])

    accuracy = float(np.mean(y_true == y_pred))
    return {
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }