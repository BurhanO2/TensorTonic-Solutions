import numpy as np

def impute_missing(X, strategy='mean'):
    """
    Fill NaN values in each feature column using column mean or median.
    """
    X = np.asarray(X, dtype=float)

    strat = np.nanmean if strategy == 'mean' else np.nanmedian
    
    # 1D input
    if X.ndim == 1:
        fill_value = strat(X)

        # If everything is NaN, use 0
        if np.isnan(fill_value):
            fill_value = 0.0

        return np.where(np.isnan(X), fill_value, X)

    # 2D input
    # strat = np.nanmean if strategy == 'mean' else np.nanmedian
    fill_values = strat(X, axis=0)

    # Replace NaN column statistics with 0
    fill_values = np.nan_to_num(fill_values, nan=0.0)

    return np.where(np.isnan(X), fill_values, X)