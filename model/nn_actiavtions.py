import numpy as np
from nn_core import Tensor, Module

class ReLU(Module):
    def forward(self, x):
        return Tensor(np.maximum(0, x.data))

class Softmax(Module):
    def forward(self, x):
        exp_x = np.exp(x.data - np.max(x.data, axis=1, keepdims=True))
        return Tensor(exp_x / np.sum(exp_x, axis=1, keepdims=True))
    
class Sigmoid:
    def forward(self, x):
        out = Tensor(1 / (1 + np.exp(-x.data)))
        
        def _backward():
            x.grad = out.grad * (out.data * (1 - out.data))
        
        out._backward = _backward
        out._parents = [x]
        return out