import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = None if requires_grad else None
        
    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor(self.data + other.data)
    
    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor(self.data * other.data)
    
    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor(self.data @ other.data)

class Module:
    def zero_grad(self):
        for p in self.parameters():
            if p.grad is not None:
                p.grad = np.zeros_like(p.data)
    
    def parameters(self):
        return []

class Linear(Module):
    def __init__(self, in_features, out_features):
        scale = np.sqrt(2.0 / in_features)
        self.weights = Tensor(
            np.random.normal(0, scale, (in_features, out_features)),
            requires_grad=True
        )
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)
    
    def forward(self, x):
        return x @ self.weights + self.bias
    
    def parameters(self):
        return [self.weights, self.bias]

class Sequential(Module):
    def __init__(self, *layers):
        self.layers = layers
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]