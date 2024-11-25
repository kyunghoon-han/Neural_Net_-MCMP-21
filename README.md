# Neural Networks Lecture Materials for MCMP-21 (Computational Physics for Master's Students in Physics, University of Luxembourg)

For the MNIST dataset, please use the link [HERE](https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/). You need to download the raw data and the CSV converter included in the link. You'll end up with two files `mnist_test.csv` and `mnist_train.csv`, where the former is for validation set and the latter for training.

## Model description

The theoretical and algorithmic descriptions of the model are given in the lecture notes included in this repository. The model introduced in this repository is made completely out of `NumPy` library for pedagogical purposes.

To change the model architecture (well... it consists only of linear layers), one can modify the following from `mnist_model.py`.

```
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
```

To run the code, run `main.py`.
