import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def layer_norm(x, eps=1e-6):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)

def vit_encoder_block(x: np.ndarray, num_heads: int,
                      Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray,
                      Wo: np.ndarray, W1: np.ndarray, W2: np.ndarray) -> np.ndarray:
    """
    Returns the float64 output of one pre-normalized ViT encoder block.
    """
    batch, tokens, width = x.shape
    d_k = width // num_heads
    normalized = layer_norm(x)

    Q = (normalized @ Wq).reshape(batch, tokens, num_heads, d_k).transpose(0, 2, 1, 3)
    K = (normalized @ Wk).reshape(batch, tokens, num_heads, d_k).transpose(0, 2, 1, 3)
    V = (normalized @ Wv).reshape(batch, tokens, num_heads, d_k).transpose(0, 2, 1, 3)

    scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(d_k)
    attention_weights = (softmax(scores) @ V).transpose(0, 2, 1, 3).reshape(batch, tokens, width)
    after_attention = x + attention_weights @ Wo
    normalized = layer_norm(after_attention)
    hidden = normalized @ W1
    gelu = 0.5 * hidden * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (hidden + 0.044715 * hidden ** 3)))

    return after_attention + gelu @ W2