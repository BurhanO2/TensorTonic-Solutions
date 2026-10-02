def local_vjp(operation: str, inputs: list, output: float, upstream_gradient: float) -> list:
    """
    Returns a list of float gradient contributions in input order.
    """
    # vals = [float(i) for i in inputs]
    
    # if operation == 'add':
    #     contributions = [upstream_gradient, upstream_gradient]
    # elif operation == 'mul':
    #     contributions = [upstream_gradient * vals[1], upstream_gradient * vals[0]]
    # else:
    #     contributions = [upstream_gradient * (1.0 - output ** 2)]

    # return contributions
    values = []
    for value in inputs:
        value = float(value)
        values.append(value)
    output = float(output)
    upstream_gradient = float(upstream_gradient)
    if operation == 'add':
        contributions = [upstream_gradient, upstream_gradient]
    elif operation == 'mul':
        contributions = [upstream_gradient * values[1], upstream_gradient * values[0]]
    else:
        contributions = [upstream_gradient * (1.0 - output ** 2)]
    return contributions
