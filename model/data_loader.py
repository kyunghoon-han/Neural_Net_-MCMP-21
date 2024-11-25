import numpy as np
import pandas as pd

def load_and_preprocess_data(filename, normalize=True):
    """Load and preprocess MNIST data from CSV file."""
    print(f"Loading data from {filename}...")
    data = pd.read_csv(filename, header=None)
    
    # Split into features and labels
    y = data.iloc[:, 0].values
    X = data.iloc[:, 1:].values
    
    # Normalize pixel values to [0, 1]
    if normalize:
        X = X / 255.0
    
    # Convert labels to one-hot encoding
    y_one_hot = np.zeros((y.size, 10))
    y_one_hot[np.arange(y.size), y] = 1
    
    return X, y_one_hot, y

class DataLoader:
    def __init__(self, X, y, batch_size=32, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(X)
    
    def __iter__(self):
        self.index = 0
        self.indices = (
            np.random.permutation(self.n_samples) 
            if self.shuffle 
            else np.arange(self.n_samples)
        )
        return self
    
    def __next__(self):
        if self.index >= self.n_samples:
            raise StopIteration
            
        batch_indices = self.indices[self.index:min(self.index + self.batch_size, self.n_samples)]
        self.index += self.batch_size
        
        return self.X[batch_indices], self.y[batch_indices]