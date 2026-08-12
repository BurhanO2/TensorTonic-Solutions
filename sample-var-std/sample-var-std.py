import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    # Write code here
    x = np.asarray(x, dtype=float)
    n = len(x)
    meanX = np.mean(x)
    var = np.sum((x - meanX) ** 2) / (n - 1)
    std = np.sqrt(var)
    return var, std