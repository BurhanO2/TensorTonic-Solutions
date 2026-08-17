import numpy as np

def stratified_split(X, y, test_size=0.2, rng=None):
    """
    Split features X and labels y into train/test while preserving class proportions.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if X.shape[0] != y.shape[0]:
        return None

    if not (0 < test_size < 1):
        return None

    rng = np.random.default_rng(42) if rng is None else rng

    train_idx = []
    test_idx = []

    classes = np.unique(y)

    for cls in classes:
        cls_idx = np.where(y == cls)[0].astype(int)

        # Shuffle within each class
        rng.shuffle(cls_idx)

        n = len(cls_idx)

        # Calculate test samples, while leaving at least
        # one sample in training when possible.
        n_test = int(round(n * test_size))
        n_test = min(n_test, n - 1)

        test_idx.extend(cls_idx[:n_test])
        train_idx.extend(cls_idx[n_test:])

    train_idx = np.sort(train_idx)
    test_idx = np.sort(test_idx)

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]