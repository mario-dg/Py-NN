from abc import abstractmethod

import numpy as np


class Loss:
    """Abstract base class representing a loss function.

    Methods
    -------
        - forward(output: np.ndarray, y: np.ndarray) -> np.ndarray
        - calculate(output: np.ndarray, y: np.ndarray) -> np.ndarray

    """

    @abstractmethod
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Forward method for computing loss.

        :param y_pred: The predicted output values as a numpy array.
        :param y_true: The target output values as a numpy array.
        :return: The computed loss as a numpy array.
        """
        raise NotImplementedError()

    def calculate(self, output: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate the loss value for a given set of predictions and ground truth labels.

        :param output: The predicted values, as a numpy array.
        :param y: The ground truth labels, as a numpy array.
        :return: The calculated loss value, as a numpy array.
        """
        sample_losses = self.forward(output, y)

        return np.mean(sample_losses)


class CategoricalCrossentropy(Loss):
    """Calculates the cross-entropy loss between predicted and true labels.

    Args:
    ----
        Loss (class): Base class for all loss functions.

    Methods:
    -------
        forward(y_pred, y_true):
            Calculates the cross-entropy loss between the predicted labels and true labels.
    """

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Forward method for CategoricalCrossentropy loss class.

        :param y_pred: Predicted values of shape (samples, classes).
        :param y_true: True values of shape (samples, classes) or (samples,) if classes are
        one-hot encoded.
        :return: Array of shape (samples,) representing the loss values for each sample.

        The forward method calculates the loss values for each sample based on the
        predicted values and true values.
        It utilizes the negative logarithm of the correct confidences of predicted values.

        Example usage:
            loss = CategoricalCrossentropy()
            result = loss.forward(y_pred, y_true)

        """
        samples = len(y_pred)

        # clip to prevent division by 0
        y_pred_clipped = np.clip(1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        else:
            raise ValueError("y_true shape not accepted.")

        return -np.log(correct_confidences)
