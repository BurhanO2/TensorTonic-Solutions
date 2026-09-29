def f1_micro(y_true: list[int], y_pred: list[int]) -> float:
    """
    Returns the micro-averaged F1 score as a Python float rounded to four decimals.
    """
    tp = sum(actual == pred for actual, pred in zip(y_true, y_pred))
    print(tp)
    error = len(y_true) - tp
    return float( 2 * tp / (2 * tp + 2 * error))