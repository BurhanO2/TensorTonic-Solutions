def random_forest_vote(predictions: list) -> list:
    """
    Returns the majority-vote label for every sample.
    """
    n = len(predictions[0])
    result = []
    for i in range(len(predictions[0])):
        votes = {}
        for j in range(len(predictions)):
            v = predictions[j][i]
            votes[v] = votes.get(v, 0) + 1
        m = max(votes.values())
        result.append(min(k for k,v in votes.items() if v == m))
    
    return result