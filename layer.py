import numpy as np


class DenseLayer:
    """A class representing a dense layer in a neural network.

    Methods
    -------
    - __init__(n_inputs, n_neurons)
    - forward(inputs)

    Attributes
    ----------
    - n_inputs (int): The number of input features to the dense layer.
    - n_neurons (int): The number of neurons in the dense layer.

    Methods
    -------
    __init__(n_inputs: int, n_neurons: int) -> None
        Initializes the DenseLayer object.

    forward(inputs: np.ndarray) -> np.ndarray
        Performs the forward pass through the dense layer.

    """

    def __init__(self, n_inputs: int, n_neurons: int) -> None:
        """Initialize a dense layer with random weights and zero biases.

        :param n_inputs: The number of inputs to the layer.
        :type n_inputs: int
        :param n_neurons: The number of neurons in the layer.
        :type n_neurons: int
        """
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Calculate output values from inputs, weights and biases.

        :param inputs: The input array to be passed forward through the dense layer.
        :return: None.
        """
        return np.dot(inputs, self.weights) + self.biases
