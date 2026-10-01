import torch
import torch.nn.functional as F

def sgns_loss(center_vec: torch.Tensor, pos_vec: torch.Tensor,
              neg_vecs: torch.Tensor) -> torch.Tensor:
    """
    Returns the scalar float64 SGNS loss.
    """
    return F.softplus(-torch.dot(center_vec, pos_vec)) + F.softplus(neg_vecs @ center_vec).sum()