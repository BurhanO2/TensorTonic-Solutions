import numpy as np

def adamw_step(w: list, m: list, v: list, grad: list, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, weight_decay: float = 0.01, eps: float = 1e-8) -> dict:
    """
    Returns a dictionary with new_w, new_m, and new_v.
    """
    # Write code here
    w = np.asarray(w, dtype=float)
    m = np.asarray(m, dtype=float)
    v = np.asarray(v, dtype=float)
    grad = np.asarray(grad, dtype=float)

    m_t = beta1 * m + (1 - beta1) * grad
    v_t = beta2 * v + (1 - beta2) * grad ** 2
    w_t = w - ((lr * m_t) / (np.sqrt(v_t) + eps)) - (lr * weight_decay * w)
    
    return {
        "new_w": w_t,
        "new_m": m_t,
        "new_v": v_t,
    }