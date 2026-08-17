import numpy as np

def kfold_split(N, k, shuffle=True, rng=None):
    """
    Returns: list of length k with tuples (train_idx, val_idx)
    """
    # Write code here
    if N <= 0 or k <= 0 or k > N:
        return None

    rng = np.random.default_rng(42) if rng is None else rng

    indices = np.arange(N)

    if shuffle:
        rng.shuffle(indices)

    folds = np.array_split(indices, k)
    result = []

    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])

        result.append((train_idx, val_idx))

    return result
