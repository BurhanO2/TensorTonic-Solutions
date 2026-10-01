import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def bert_fine_tuning_step(hidden_states: np.ndarray, labels: np.ndarray,
                          classifier_W: np.ndarray, classifier_b: np.ndarray,
                          learning_rate: float) -> dict:
    """
    Returns updated classifier parameters and the pre-update loss.
    """
    z = hidden_states[:, 0, :] @ classifier_W + classifier_b
    probs = softmax(z)
    batch_size = labels.shape[0]
    grad_z = probs.copy()
    grad_z[np.arange(batch_size), labels] -= 1.0
    grad_z /= batch_size
    grad_W = hidden_states[:, 0, :].T @ grad_z
    grad_b = np.sum(grad_z, axis=0)
    loss = -np.mean(np.log(probs[np.arange(batch_size), labels]))

    return {
        "new_classifier_W": classifier_W - learning_rate * grad_W,
        "new_classifier_b": classifier_b - learning_rate * grad_b,
        "loss": float(loss),
    }