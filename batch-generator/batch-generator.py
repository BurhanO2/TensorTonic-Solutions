import numpy as np

def batch_generator(X, y, batch_size, rng=None, drop_last=False):
    """
    Randomly shuffle a dataset and yield mini-batches (X_batch, y_batch).
    """
    X = np.asarray(X)
    y = np.asarray(y)

    rng = np.random.default_rng(seed=42) if rng is None else rng

    indices = rng.permutation(len(X))

    for start in range(0, len(X), batch_size):
        end = start + batch_size

        if drop_last and end > len(X):
            break

        batch_indices = indices[start:end]

        yield X[batch_indices], y[batch_indices]