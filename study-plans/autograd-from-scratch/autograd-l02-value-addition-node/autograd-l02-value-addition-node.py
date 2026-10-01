def value_addition_node(left: dict, right: dict, output_id: str) -> dict:
    """
    Returns an addition node with id, data, grad, op, and the original parents.
    """
    return {
        'id': output_id,
        'data': float(left['data'] + right['data']),
        'grad': 0.0,
        'op': '+',
        'parents': [left, right]
    }