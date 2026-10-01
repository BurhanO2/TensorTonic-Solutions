import numpy as np

def gradient_check_product_chain(
    a: float,
    b: float,
    c: float,
    f: float,
    h: float,
) -> tuple[float, list, list, float]:
    """
    Returns loss, analytic gradients, numerical gradients, and maximum error.
    """
    def loss(a, b, c, f):
        return float((a * b + c) * f)

    base_loss = loss(a, b, c, f)
    intermediate = a * b + c
    analytic = [b * f, a * f, f, intermediate]
    numerical = [(loss(a + h, b, c, f) - base_loss) / h, (loss(a, b + h, c, f) - base_loss) / h, (loss(a, b, c + h, f) - base_loss) / h, (loss(a, b, c, f + h) - base_loss) / h]
    max_error = float(np.max(np.abs(np.asarray(analytic) - np.asarray(numerical))))
    return (base_loss, [float(value) for value in analytic], [float(value) for value in numerical], max_error)
