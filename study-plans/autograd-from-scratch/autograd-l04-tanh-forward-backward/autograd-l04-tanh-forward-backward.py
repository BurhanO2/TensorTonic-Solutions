import torch

def tanh_forward_backward(x: torch.Tensor, upstream_gradient: torch.Tensor) -> tuple:
    """
    Returns a tuple of scalar tensors: tanh output and input gradient.
    """
    output = torch.tanh(x)
    input_gradient = upstream_gradient * (1 - output.square())
    return output, input_gradient
