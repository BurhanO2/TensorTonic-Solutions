import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def gru_cell_forward(x: list, h_prev: list, params: dict) -> np.ndarray:
    """
    Returns the updated hidden state as a NumPy array matching the shape of h_prev.
    """
    # Write code here
    x = np.asarray(x, dtype=float)
    h_prev = np.asarray(h_prev, dtype=float)
    parameters = {name: np.asarray(value, dtype=float) for name, value in params.items()}
    
    singular = x.ndim == 1
    if singular:
        x = x.reshape(1, -1)
        h_prev = h_prev.reshape(1, -1)

    z_t = sigmoid(x @ parameters["Wz"] + h_prev @ parameters["Uz"] + parameters["bz"])
    r_t = sigmoid(x @ parameters["Wr"] + h_prev @ parameters["Ur"] + parameters["br"])
    h_tilde = np.tanh(x @ parameters["Wh"] + (r_t * h_prev) @ parameters["Uh"] + parameters["bh"])
    h_t = (1 - z_t) * h_prev + z_t * h_tilde
    return h_t[0] if singular else h_t
