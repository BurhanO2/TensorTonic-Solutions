import torch

def neuron_gradient_check(inputs: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor, h: float) -> tuple:
    """
    Returns five tensors: analytic/numerical weight gradients, analytic/numerical bias gradients, maximum error.
    """
    x, w, b = inputs, weights, bias
    y = torch.tanh(torch.sum(x * w) + b)

    grad_b = 1 - y.square()
    grad_w = (1 - y.square()) * x

    grad_num = torch.empty_like(w)

    for i in range(w.numel()):
        pertub_w = w.clone()
        pertub_w[i] += h
        pertub_o = torch.tanh(torch.sum(x * pertub_w) + b)
        grad_num[i] = (pertub_o - y) / h

    grad_a = grad_b
    grad_num_b = (torch.tanh(torch.sum(x * w) + b + h) - y) / h

    bias_error = torch.abs(grad_a - grad_num_b)

    if w.numel() == 0:
        max_error = bias_error
    else:
        weight_error = torch.max(torch.abs(grad_w - grad_num))
        max_error = torch.maximum(weight_error, bias_error)

    return grad_w, grad_num, grad_a, grad_num_b, max_error