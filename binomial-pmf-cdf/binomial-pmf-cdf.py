import numpy as np
from scipy.special import comb

def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial PMF and CDF.
    """
    # Write code here
    def get_pmf(i):
        return comb(n, i) * (p ** i) * ((1 - p) ** (n - i))

    pmf = get_pmf(k)
    cdf = sum(get_pmf(i) for i in range(k + 1))

    return float(pmf), float(cdf)