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
        if (
            not isinstance(n_inputs, int)
            or not isinstance(n_neurons, int)
            or n_inputs <= 0
            or n_neurons <= 0
        ):
            raise ValueError("n_inputs and n_neurons must be positive integers")

        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward method of DenseLayer class.

        :param inputs: Input data for the dense layer. Should be a numpy ndarray.
        :return: Output data after the forward pass through the dense layer.
                 Returns a numpy ndarray.

        """
        if inputs.shape[1] != self.weights.shape[0]:
            raise ValueError('The number of columns in inputs must be equal to the number of rows in weights.')
        return np.dot(inputs, self.weights) + self.biases
