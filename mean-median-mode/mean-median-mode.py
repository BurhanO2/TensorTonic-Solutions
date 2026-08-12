import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    # Write code here
    x = np.asarray(x)
    
    mean = float(np.mean(x))
    
    median = float(np.median(x))
    
    values, counts = np.unique(x, return_counts=True)
    mode_ix = counts.argmax()
    mode = float(values[mode_ix])

    return (mean, median, mode)