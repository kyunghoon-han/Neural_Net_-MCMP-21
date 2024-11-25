from nn_core import Sequential, Linear
from nn_actiavtions import ReLU, Softmax

class MNISTClassifier(Sequential):
    def __init__(self):
        super().__init__(
            Linear(784, 256),
            ReLU(),
            Linear(256, 32),
            ReLU(),
            Linear(32, 10),
            Softmax()
        )