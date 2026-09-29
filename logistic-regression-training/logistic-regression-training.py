import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here
    N, D = X.shape
    w = np.zeros(D)
    b = 0.0
    for _ in range(steps):
        logits = X @ w + b
        preds = _sigmoid(logits)
        w_hat = X.T @ (preds - y) / N
        w -= lr * w_hat
        b_hat = np.mean(preds - y)
        b -= lr * b_hat
    return w, b