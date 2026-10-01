import numpy as np

def batch_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=0)
    var = x.var(axis=0)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta

def batch_norm_block(x, W1, W2, gamma1, beta1, gamma2, beta2, mode):
    """
    Returns the normalized residual-block result and selected mode in a dictionary.
    """
    x = np.array(x, dtype=float)
    W1 = np.array(W1, dtype=float)
    W2 = np.array(W2, dtype=float)
    gamma1 = np.array(gamma1, dtype=float)
    beta1 = np.array(beta1, dtype=float)
    gamma2 = np.array(gamma2, dtype=float)
    beta2 = np.array(beta2, dtype=float)
    i = x.copy()

    if mode == "post":
        output = np.maximum(0, batch_norm(x @ W1, gamma1, beta1))
        output = np.maximum(0, batch_norm(output @ W2, gamma2, beta2) + x)
        return {
            "output": output,
            "mode": "post"
        }
    else:
        output = np.maximum(0, batch_norm(x, gamma1, beta1)) @ W1
        output = np.maximum(0, batch_norm(output, gamma2, beta2)) @ W2 + x
        return {
            "output": output,
            "mode": "pre"
        }