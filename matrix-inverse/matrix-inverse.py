import numpy as np

def matrix_inverse(A):
    """
    Returns: A_inv of shape (n, n) such that A @ A_inv ≈ I
    """
    # Write code here
    A = np.asarray(A, dtype=float)

    if A.ndim != 2:
        return None
    if A.shape[0] != A.shape[1]:
        return None
    if abs(np.linalg.det(A)) < np.exp(-10):
        return None

    A_inv = np.linalg.inv(A)
    return A_inv
