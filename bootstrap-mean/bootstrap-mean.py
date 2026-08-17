import numpy as np

def bootstrap_mean(x, n_bootstrap=1000, ci=0.95, rng=None):
    """
    Returns: (boot_means, lower, upper)
    """
    # Write code here
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        return None

    rng = np.random.default_rng(seed=42) if rng is None else rng
    alpha = float(1 - ci)

    samples = rng.choice(x, size=(n_bootstrap, len(x)), replace=True)
    boot_means = samples.mean(axis=1)

    lower = np.quantile(boot_means, alpha / 2)
    upper = np.quantile(boot_means, 1 - alpha / 2)

    return boot_means, lower, upper
