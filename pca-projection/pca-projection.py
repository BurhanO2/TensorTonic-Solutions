import numpy as np

def pca_projection(X, k):
    """
    Project data onto the top-k principal components.
    """
    # Write code here
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        return None
    if k <= 0 or k > X.shape[1]:
        return None

    X_centered = X - np.mean(X, axis=0)
    cov = np.cov(X_centered, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    indices = np.argsort(eigenvalues)[::-1]
    components = eigenvectors[:, indices[:k]]

    return X_centered @ components
