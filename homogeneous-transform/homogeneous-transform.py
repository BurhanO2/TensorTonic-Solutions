import numpy as np

def apply_homogeneous_transform(T: list, points: list) -> np.ndarray:
    """
    Returns transformed points with shape (3,) or (N, 3).
    """
    # Write code here
    T = np.asarray(T, dtype=float)
    points = np.asarray(points, dtype=float)
    single_point = points.ndim == 1

    if single_point:
        points = points.reshape(1, 3)

    ones = np.ones((points.shape[0], 1))
    points_h = np.concatenate([points, ones], axis=1)
    transformed = (T @ points_h.T).T[:, :3]

    if single_point:
        return transformed[0]
    else:
        return transformed
