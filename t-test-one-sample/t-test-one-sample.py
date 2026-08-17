import numpy as np

def t_test_one_sample(x, mu0):
    """
    Compute one-sample t-statistic.
    """
    # Write code here
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = np.mean(x)

    s = np.sqrt((1 / (n - 1)) * np.sum((x - mean) ** 2))

    t = (mean - mu0) / (s / np.sqrt(n))
    return t