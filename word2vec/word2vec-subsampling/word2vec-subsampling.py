import torch

def subsample_keep_probs(counts: torch.Tensor,
                         t: float = 1e-5) -> torch.Tensor:
    """
    Returns the float64 keep probability for every vocabulary word.
    """
    freq = counts / counts.sum()
    return torch.clamp(torch.sqrt(t / freq), max=1.0)