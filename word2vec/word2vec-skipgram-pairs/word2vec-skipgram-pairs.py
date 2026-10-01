import torch

def skipgram_pairs(token_ids: torch.Tensor, window: int) -> torch.Tensor:
    """
    Returns the ordered center-context pairs as an int64 tensor.
    """
    pairs = []
    for i, token in enumerate(token_ids.tolist()):
        start = max(0, i - window)
        stop = min(len(token_ids), i + window + 1)
        for context_index in range(start, stop):
            if context_index != i:
                pairs.append([token, int(token_ids[context_index])])

    return torch.tensor(pairs, dtype=torch.int64).reshape(-1, 2)