import numpy as np
from tqdm import tqdm
from nn_core import Tensor

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(pred, target):
    n_samples = pred.shape[0]
    log_likelihood = -np.log(pred[range(n_samples), target.argmax(axis=1)])
    return np.sum(log_likelihood) / n_samples

class Trainer:
    def __init__(self, model, learning_rate=0.01):
        self.model = model
        self.learning_rate = learning_rate
    
    def _compute_gradients(self, X, y):
        batch_size = X.shape[0]
        
        # Forward pass through layers
        h1 = X @ self.model.layers[0].weights.data + self.model.layers[0].bias.data
        a1 = np.maximum(0, h1)  # ReLU
        h2 = a1 @ self.model.layers[2].weights.data + self.model.layers[2].bias.data
        a2 = np.maximum(0, h2)  # ReLU
        logits = a2 @ self.model.layers[4].weights.data + self.model.layers[4].bias.data
        probs = softmax(logits)
        
        # Backward pass
        grad_y = probs.copy()
        grad_y[range(batch_size), y.argmax(axis=1)] -= 1
        grad_y /= batch_size
        
        # Last layer
        grad_w3 = a2.T @ grad_y
        grad_b3 = np.sum(grad_y, axis=0)
        grad_a2 = grad_y @ self.model.layers[4].weights.data.T
        
        # Second layer
        grad_h2 = grad_a2 * (h2 > 0)  # ReLU derivative
        grad_w2 = a1.T @ grad_h2
        grad_b2 = np.sum(grad_h2, axis=0)
        grad_a1 = grad_h2 @ self.model.layers[2].weights.data.T
        
        # First layer
        grad_h1 = grad_a1 * (h1 > 0)  # ReLU derivative
        grad_w1 = X.T @ grad_h1
        grad_b1 = np.sum(grad_h1, axis=0)
        
        return [(grad_w1, grad_b1), (grad_w2, grad_b2), (grad_w3, grad_b3)]
    
    def train_epoch(self, train_loader, desc="Training"):
        total_loss = 0
        correct = 0
        n_samples = 0
        
        for X, y in tqdm(train_loader, desc=desc):
            batch_size = X.shape[0]
            n_samples += batch_size
            
            # Forward pass
            h1 = X @ self.model.layers[0].weights.data + self.model.layers[0].bias.data
            a1 = np.maximum(0, h1)
            h2 = a1 @ self.model.layers[2].weights.data + self.model.layers[2].bias.data
            a2 = np.maximum(0, h2)
            logits = a2 @ self.model.layers[4].weights.data + self.model.layers[4].bias.data
            probs = softmax(logits)
            
            # Compute loss and accuracy
            loss = cross_entropy_loss(probs, y)
            predictions = np.argmax(probs, axis=1)
            correct += np.sum(predictions == np.argmax(y, axis=1))
            total_loss += loss * batch_size
            
            # Compute gradients
            gradients = self._compute_gradients(X, y)
            
            # Update weights
            layer_indices = [0, 2, 4]  # Indices of Linear layers
            for idx, (grad_w, grad_b) in zip(layer_indices, gradients):
                self.model.layers[idx].weights.data -= self.learning_rate * grad_w
                self.model.layers[idx].bias.data -= self.learning_rate * grad_b
        
        return total_loss / n_samples, correct / n_samples
    
    def evaluate(self, val_loader):
        correct = 0
        n_samples = 0
        
        for X, y in val_loader:
            n_samples += X.shape[0]
            
            # Forward pass
            h1 = X @ self.model.layers[0].weights.data + self.model.layers[0].bias.data
            a1 = np.maximum(0, h1)
            h2 = a1 @ self.model.layers[2].weights.data + self.model.layers[2].bias.data
            a2 = np.maximum(0, h2)
            logits = a2 @ self.model.layers[4].weights.data + self.model.layers[4].bias.data
            probs = softmax(logits)
            
            predictions = np.argmax(probs, axis=1)
            correct += np.sum(predictions == np.argmax(y, axis=1))
        
        return correct / n_samples