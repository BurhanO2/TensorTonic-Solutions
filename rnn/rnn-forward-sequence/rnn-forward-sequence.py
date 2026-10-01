import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray, W_xh: np.ndarray,
                W_hh: np.ndarray, b_h: np.ndarray) -> dict:
    """
    Returns hidden_states and final_hidden_state as float64 arrays.
    """
    h_t = h_0.copy()
    f_h_t = []
    for step in range(X.shape[1]):
        h_t = np.tanh(X[:, step, :] @ W_xh.T + h_t @ W_hh.T + b_h)
        f_h_t.append(h_t)

    return {
        "hidden_states": np.stack(f_h_t, axis=1),
        "final_hidden_state": h_t
    }