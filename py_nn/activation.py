import numpy as np


def relu(inputs: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit (ReLU) activation function.

    :param inputs: Input array.
    :type inputs: np.ndarray
    :return: Output array after applying ReLU activation function.
    :rtype: np.ndarray
    """
    return np.clip(inputs, 0, None)


def softmax(inputs: np.ndarray) -> np.ndarray:
    """Softmax activation function.

    :param inputs: Input array.
    :type inputs: np.ndarray
    :return: Output array after applying Softmax activation function.
    :rtype: np.ndarray
    """
    # Un-normalized probabilities
    exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

    # Normalize per sample
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)
