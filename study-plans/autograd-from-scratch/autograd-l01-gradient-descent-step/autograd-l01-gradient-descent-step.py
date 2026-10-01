import numpy as np

def gradient_descent_step(values: list, gradients: list, learning_rate: float) -> tuple[list, float]:
    """
    Returns a fresh list of updated values and the predicted objective change.
    """
    values_array = np.asarray(values, dtype=np.float64)
    gradients_array = np.asarray(gradients, dtype=np.float64)
    rate = float(learning_rate)
    updated = values_array - rate * gradients_array
    predicted_change = float(np.dot(gradients_array, updated - values_array))
    return ([float(value) for value in updated], predicted_change)
    # values = np.asarray(values, dtype=float)
    # gradients = np.asarray(gradients, dtype=float)
    # theta = values - learning_rate * gradients
    # change = np.dot(gradients_array, updated - values_array)

    # return (list(theta), float(change))
