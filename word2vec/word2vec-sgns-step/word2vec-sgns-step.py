import torch

def sgns_sgd_step(W_in: torch.Tensor, W_out: torch.Tensor,
                  center_id: int, pos_id: int,
                  neg_ids: torch.Tensor, lr: float) -> dict:
    """
    Returns updated W_in and W_out float64 tensors in a dictionary.
    """
    new_W_in = W_in.clone()
    new_W_out = W_out.clone()
    center = W_in[center_id].clone()
    output = W_out.clone()

    positive_coeff = torch.sigmoid(torch.dot(center, output[pos_id])) - 1.0
    center_grad = positive_coeff * output[pos_id]
    output_grad = {pos_id: positive_coeff * center}

    for neg_id in neg_ids.tolist():
        coeff = torch.sigmoid(torch.dot(center, output[neg_id]))
        center_grad = center_grad + coeff * output[neg_id]
        output_grad[neg_id] = output_grad.get(neg_id, torch.zeros_like(center)) + coeff * center

    W_in[center_id] = W_in[center_id] - lr * center_grad
    
    for id, grad in output_grad.items():
        W_out[id] = W_out[id] - lr * grad

    return {
        "W_in": W_in,
        "W_out": W_out
    }