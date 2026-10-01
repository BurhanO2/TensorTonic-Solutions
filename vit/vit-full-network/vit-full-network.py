import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def layer_norm(x, eps=1e-6):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)

def vit_forward(image: np.ndarray, patch_size: int, num_heads: int,
                W_patch: np.ndarray, patch_bias: np.ndarray,
                cls_token: np.ndarray, pos_embed: np.ndarray,
                encoder_weights: list, W_head: np.ndarray) -> np.ndarray:
    """
    Returns float64 Vision Transformer logits with shape (B, C).
    """
    batch, height, width, channels = image.shape
    h = height // patch_size
    w = width // patch_size
    usable = image[:, :h * patch_size, :w * patch_size, :]
    patches = usable.reshape(batch, h, patch_size, w, patch_size, channels)
    patches = patches.transpose(0, 1, 3, 2, 4, 5).reshape(batch, h * w, -1)
    tokens = patches @ W_patch + patch_bias
    tokens = np.concatenate((np.broadcast_to(cls_token, (batch, 1, W_patch.shape[1])), tokens), axis=1)
    tokens = tokens + pos_embed

    for weights in encoder_weights:
        normalized = layer_norm(tokens)
        d_k = tokens.shape[-1] // num_heads
        Q = (normalized @ weights["Wq"]).reshape(batch, -1, num_heads, d_k).transpose(0, 2, 1, 3)
        K = (normalized @ weights["Wk"]).reshape(batch, -1, num_heads, d_k).transpose(0, 2, 1, 3)
        V = (normalized @ weights["Wv"]).reshape(batch, -1, num_heads, d_k).transpose(0, 2, 1, 3)
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(d_k)
        attended = (softmax(scores) @ V).transpose(0, 2, 1, 3).reshape(batch, -1, tokens.shape[-1])
        tokens = tokens + attended @ weights["Wo"]
        normalized = layer_norm(tokens)
        hidden = normalized @ weights["W1"]
        gelu = 0.5 * hidden * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (hidden + 0.044715 * hidden ** 3)))
        tokens = tokens + gelu @ weights["W2"]

    return layer_norm(tokens[:, 0, :]) @ W_head