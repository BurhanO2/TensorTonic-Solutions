import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
        
    _, counts = np.unique(y, return_counts=True)
    p = counts / y.size
    
    return -np.sum(p * np.log2(p))