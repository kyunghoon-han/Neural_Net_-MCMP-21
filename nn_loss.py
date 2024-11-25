import numpy as np
from nn_core import Tensor

class CrossEntropyLoss:
    def __call__(self, pred, target):
        eps = 1e-12
        pred_clipped = np.clip(pred.data, eps, 1 - eps)
        loss = -np.sum(target.data * np.log(pred_clipped)) / pred.data.shape[0]
        out = Tensor(loss)
        
        def _backward():
            # Gradient of cross entropy
            if pred.requires_grad:
                pred.grad = (pred_clipped - target.data) / pred.data.shape[0]
            
        out._backward = _backward
        out._parents = [pred]
        return out