def value_multiplication_node(left: dict, right: dict, output_id: str) -> dict:
    """
    Returns the node dictionary with id, data, grad, op, and parents.
    """
    return {
        'id': output_id,
        'data': float(left['data'] * right['data']),
        'grad': 0.0,
        'op': '*',
        'parents': [left, right]
    }
