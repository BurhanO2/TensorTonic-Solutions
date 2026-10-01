import torch

def neuron_backward(inputs: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor, upstream_gradient: torch.Tensor) -> tuple:
    """
    Returns a tuple of tensors: output, input gradients, weight gradients, bias gradient.
    """
    x = inputs
    w = weights
    b = bias
    upstream = upstream_gradient
    output = torch.tanh(torch.sum(x * w) + b)
    local_gradient = upstream * (1 - output.square())
    input_gradients = local_gradient * w
    weight_gradients = local_gradient * x
    bias_gradient = local_gradient
    return (output, input_gradients, weight_gradients, bias_gradient)
    # x, w, b = inputs, weights, bias
    # y = torch.tanh(torch.sum(x * w + b))
    # delta = upstream_gradient * (1 - y.square())
    # grad_x = delta * w
    # grad_w = delta * x
    # grad_b = delta

    # return y, grad_x, grad_w, grad_b