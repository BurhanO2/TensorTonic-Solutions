import numpy as np
import math

def poisson_pmf_cdf(lam, k):
    """
    Compute Poisson PMF and CDF.
    """
    # Write code here
    def calc_pmf(i):
        return (np.exp(-lam) * (lam ** i)) / math.factorial(i)

    pmf = calc_pmf(k)
    cdf = np.sum([calc_pmf(i) for i in range(0, k + 1)])
    return pmf, cdf