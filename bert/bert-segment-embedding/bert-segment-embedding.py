import numpy as np

def bert_embeddings(token_ids: np.ndarray, segment_ids: np.ndarray,
                    token_embeddings: np.ndarray, position_embeddings: np.ndarray,
                    segment_embeddings: np.ndarray) -> np.ndarray:
    """
    Returns the float64 BERT input embeddings with shape (B, S, H).
    """
    seq_length = token_ids.shape[1]
    pos_values = position_embeddings[np.arange(seq_length)][None, :, :]
    return token_embeddings[token_ids] + pos_values + segment_embeddings[segment_ids]