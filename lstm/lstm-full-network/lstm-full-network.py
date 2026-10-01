import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def lstm_forward(X: np.ndarray, W_f: np.ndarray, W_i: np.ndarray,
                 W_c: np.ndarray, W_o: np.ndarray, b_f: np.ndarray,
                 b_i: np.ndarray, b_c: np.ndarray, b_o: np.ndarray,
                 W_y: np.ndarray, b_y: np.ndarray) -> dict:
    """
    Returns outputs, final_hidden_state, and final_cell_state.
    """
    batch, steps, _ = X.shape
    h_t = np.zeros((batch, b_f.shape[0]), dtype=np.float64)
    C_t = np.zeros_like(h_t)
    outputs = []
    for step in range(steps):
        joined = np.concatenate([h_t, X[:, step, :]], axis=-1)
        f_t = sigmoid(joined @ W_f.T + b_f)
        i_t = sigmoid(joined @ W_i.T + b_i)
        C_tilde = np.tanh(joined @ W_c.T + b_c)
        o_t = sigmoid(joined @ W_o.T + b_o)
        C_t = f_t * C_t + i_t * C_tilde
        h_t = o_t * np.tanh(C_t)
        outputs.append(h_t @ W_y.T + b_y)

    return {
        "outputs": np.stack(outputs, axis=1),
        "final_hidden_state" : h_t,
        "final_cell_state" : C_t
    } 
