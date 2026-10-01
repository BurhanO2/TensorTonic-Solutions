def scalar_expression_partials(a: float, b: float, c: float, h: float) -> tuple[float, float, float, float]:
    """
    Returns the expression value and numerical partials for a, b, and c.
    """
    def expression(a, b, c):
        return float(a * b + c)

    d = expression(a, b, c)
    partial_a = (expression(a + h, b, c) - d) / h
    partial_b = (expression(a, b + h, c) - d) / h
    partial_c = (expression(a, b, c + h) - d) / h
    return (d, float(partial_a), float(partial_b), float(partial_c))
