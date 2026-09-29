import numpy as np

def conv2d(x: list, W: list, b: list) -> np.ndarray:
    """
    Returns the convolved batch as a floating-point NumPy array.
    """
    # Write code here
    x = np.asarray(x, dtype=float)
    W = np.asarray(W, dtype=float)
    b = np.asarray(b, dtype=float)

    batch_size, _, height, width = x.shape
    output_channels, _, kernel_h, kernel_w = W.shape
    output_height = height - kernel_h + 1
    output_width = width - kernel_w + 1
    output = np.zeros((batch_size, output_channels, output_height, output_width), dtype=float)

    for n in range(batch_size):
        for output_channel in range(output_channels):
            for row in range(output_height):
                for col in range(output_width):
                    patch = x[n, :, row : row + kernel_h, col : col + kernel_w]
                    output[n, output_channel, row, col] = np.sum(patch * W[output_channel]) + b[output_channel]
    return output