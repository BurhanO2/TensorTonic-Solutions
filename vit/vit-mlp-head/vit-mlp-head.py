import numpy as np

def classification_head(encoder_output: np.ndarray,
                        W_head: np.ndarray) -> np.ndarray:
    """
    Returns float64 class logits with shape (B, C).
    """
    cls_state = encoder_output[:, 0, :]
    mean = cls_state.mean(axis=-1, keepdims=True)
    var = cls_state.var(axis=-1, keepdims=True)
    normalized = (cls_state - mean) / np.sqrt(var + 1e-6)
    return normalized @ W_head
