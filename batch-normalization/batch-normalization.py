import numpy as np

def batch_norm_forward(x: list, gamma: list, beta: list, eps: float = 1e-5) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    # Write code here
    x = np.asarray(x, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    beta = np.asarray(beta, dtype=float)

    if x.ndim == 2:
        axes = (0,)
        param_shape = (1, -1)
    else:
        axes = (0,2,3)
        param_shape = (1, -1, 1, 1)
    mean = np.mean(x, axis=axes, keepdims=True)
    var = np.var(x, axis=axes, keepdims=True)
    norm = (x - mean) / np.sqrt(var + eps)
    return norm * gamma.reshape(param_shape) + beta.reshape(param_shape)