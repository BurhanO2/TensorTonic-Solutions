import numpy as np

def finite_difference_derivative(coefficients: list, x: float, h: float) -> tuple[float, float, float]:
    """
    Returns the value at x, the value at x plus h, and the estimated slope.
    """
    values = np.asarray(coefficients, dtype=float)

    def evaluate(point):
        result = 0.0
        for coefficient in values[::-1]:
            result = result * point + coefficient
    
        return float(result)
                        
    slope = (evaluate(x + h) - evaluate(x)) / h
    return (evaluate(x), evaluate(x + h), float(slope))
