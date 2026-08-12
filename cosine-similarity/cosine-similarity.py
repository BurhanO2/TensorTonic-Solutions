import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    # Write code here
    dot = np.dot(a, b)
    normA = np.linalg.norm(a)
    normB = np.linalg.norm(b)

    if normA == 0 or normB == 0:
        return 0.0

    return float(dot / (normA * normB))