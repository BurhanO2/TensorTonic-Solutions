import math
from collections import Counter

def bleu_score(candidate: list, reference: list, max_n: int) -> float:
    """
    Returns the unsmoothed BLEU score.
    """
    # Write code here
    c = len(candidate)

    if c == 0:
        return 0.0

    r = len(reference)
    bp = math.exp(1 - r / c)  if c < r else 1.0
    precisions = []

    for n in range(1, max_n + 1):
        candidate_ngrams = []
        for i in range(c - n + 1):
            candidate_ngrams.append(tuple(candidate[i : i + n]))
        
        reference_ngrams = []
        for i in range(r - n + 1):
            reference_ngrams.append(tuple(reference[i : i + n]))

        if len(candidate_ngrams) == 0:
            precisions.append(0.0)
            continue
        
        cc = Counter(candidate_ngrams)
        rc = Counter(reference_ngrams)
        clipped = 0
        for ngram, count in cc.items():
            clipped += min(count, rc.get(ngram, 0))
        
        precisions.append(clipped / len(candidate_ngrams))
        
    for p in precisions:
        if p == 0:
            return 0.0

    log_mean = sum(math.log(p) for p in precisions) / len(precisions) 
    return bp * math.exp(log_mean)