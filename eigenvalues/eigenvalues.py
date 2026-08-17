import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """
    # Write code here
    try:
        matrix = np.asarray(matrix, dtype=float)
    except (ValueError, TypeError):
        return None
    if matrix.ndim != 2:
        return None
    if matrix.shape[0] != matrix.shape[1]:
        return None
    eigen_values = np.linalg.eigvals(matrix)
    idx = np.lexsort((eigen_values,))
    eigen_values = eigen_values[idx]
    print(eigen_values)
    return eigen_values