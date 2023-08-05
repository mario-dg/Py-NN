import nnfs
from nnfs.datasets import spiral_data

import activation as a
from layer import DenseLayer

nnfs.init()

X, y = spiral_data(samples=100, classes=3)

dense1 = DenseLayer(2, 3)
dense2 = DenseLayer(3, 3)

X = a.relu(dense1.forward(X))

X = a.softmax(dense2.forward(X))

print(X[:5])
