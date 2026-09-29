import numpy as np

def kl_divergence(p: list, q: list, eps: float = 1e-12) -> float:
    """
    Returns the divergence as a float.
    """
    # Write code here
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    pos = p > 0
    q_safe = np.clip(q[pos], eps, None)
    return float(np.sum(p[pos] * np.log(p[pos] / q_safe)))