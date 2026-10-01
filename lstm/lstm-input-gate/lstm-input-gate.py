import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def input_gate(h_prev: np.ndarray, x_t: np.ndarray,
               W_i: np.ndarray, b_i: np.ndarray,
               W_c: np.ndarray, b_c: np.ndarray) -> dict:
    """
    Returns input_gate and candidate_state as float64 arrays.
    """
    i_t = sigmoid(np.concatenate([h_prev, x_t], axis=-1) @ W_i.T + b_i)
    c_t = np.tanh(np.concatenate([h_prev, x_t], axis=-1) @ W_c.T + b_c)
    return {
        "input_gate": i_t,
        "candidate_state": c_t
    }