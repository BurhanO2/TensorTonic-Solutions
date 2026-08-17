import numpy as np
def k_means_assignment(points, centroids):
    """
    Assign each point to the nearest centroid.
    """
    # Write code here
    points = np.asarray(points)
    centroids = np.asarray(centroids)
    assignments = []
    for i in range(len(points)):
        min = 1000000
        ix = -1
        for j in range(len(centroids)):
            dist = np.linalg.norm(points[i] - centroids[j])
            if dist < min:
                min = dist
                ix = j
        assignments.append(ix)

    return assignments