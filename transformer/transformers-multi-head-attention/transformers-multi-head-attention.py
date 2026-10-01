import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Returns projected multi-head attention outputs.
    """
    batch_size, seq_len, d_model = Q.shape
    d_k = d_model // num_heads

    Q = (Q @ W_q).reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    K = (K @ W_k).reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    V = (V @ W_v).reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)

    scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(d_k)
    attention_output = softmax(scores) @ V
    attention_output = attention_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
    return attention_output @ W_o