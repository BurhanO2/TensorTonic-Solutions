import numpy as np

def linear_regression_closed_form(X, y):
    """
    Compute the optimal weight vector using the normal equation.
    """
    # Write code here
    X = np.asarray(X)
    y = np.asarray(y)

    inv = np.linalg.inv(X.T @ X)
    w = np.asarray(inv @ X.T @ y, dtype=float)

    return w