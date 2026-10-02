def local_vjp(operation: str, inputs: list, output: float, upstream_gradient: float) -> list:
    """
    Returns a list of float gradient contributions in input order.
    """
    output = float(output)
    upstream_gradient = float(upstream_gradient)

    if operation == 'add':
        contributions = [upstream_gradient, upstream_gradient]
    elif operation == 'mul':
        contributions = [upstream_gradient * inputs[1], upstream_gradient * inputs[0]]
    else:
        contributions = [upstream_gradient * (1.0 - output ** 2)]

    return contributions
