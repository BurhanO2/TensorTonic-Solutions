def k_means_centroid_update(points: list, assignments: list, k: int) -> list:
    """
    Returns one updated centroid for each cluster.
    """
    # Write code here
    dims = len(points[0])
    sums = [[0.0] * dims for _ in range(k)]
    counts = [0] * k
    for i, j in enumerate(points):
        c = assignments[i]
        counts[c] += 1
        for d in range(dims):
            sums[c][d] += j[d]

    result = []
    for x in range(k):
      row = []
      for d in range(dims):
        if counts[x] > 0:
          val = sums[x][d] / counts[x]
        else:
          val = 0.0
        row.append(val)
      result.append(row)
    
    return result
        
            