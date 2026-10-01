import numpy as np

def patch_embed(image: np.ndarray, patch_size: int,
                W_proj: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    Returns float64 patch embeddings with shape (B, N, D).
    """
    batch, height, width, channels = image.shape
    h =  height // patch_size
    w =  width // patch_size

    patch = image[:, :h * patch_size, :w * patch_size, :]
    x_p = patch.reshape(batch, h, patch_size, w, patch_size, channels)
    x_p = x_p.transpose(0, 1, 3, 2, 4, 5)
    x_p = x_p.reshape(batch, h * w, -1)
    return x_p @ W_proj + bias