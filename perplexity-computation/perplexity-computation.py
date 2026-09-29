import math

def perplexity(prob_distributions: list, actual_tokens: list) -> float:
    """
    Returns the sequence perplexity.
    """
    # Write code here
    n = len(actual_tokens)
    log_sum = 0.0
    for i in range(n):
        p = prob_distributions[i][actual_tokens[i]]
        log_sum += math.log(p)

    H = -log_sum / n
    return round(math.exp(H), 4)