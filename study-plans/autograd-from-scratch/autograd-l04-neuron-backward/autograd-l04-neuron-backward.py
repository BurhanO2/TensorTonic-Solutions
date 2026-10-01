import torch

def neuron_backward(inputs: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor, upstream_gradient: torch.Tensor) -> tuple:
    """
    Returns a tuple of tensors: output, input gradients, weight gradients, bias gradient.
    """
    x, w, b = inputs, weights, bias
    y = torch.tanh(torch.sum(x * w) + b)
    delta = upstream_gradient * (1 - y.square())
    input_gradients = delta * w
    weight_gradients = delta * x
    bias_gradient = delta

    return (y, input_gradients, weight_gradients, bias_gradient)
    # x, w, b = inputs, weights, bias
    # y = torch.tanh(torch.sum(x * w + b))
    # delta = upstream_gradient * (1 - y.square())
    # grad_x = delta * w
    # grad_w = delta * x
    # grad_b = delta

    # return y, grad_x, grad_w, grad_b