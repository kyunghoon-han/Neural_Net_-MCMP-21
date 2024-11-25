from data_loader import load_and_preprocess_data, DataLoader
from mnist_model import MNISTClassifier
from trainer import Trainer
import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def plot_sample_predictions(model, X, y, epoch):
    """Plot a grid of sample predictions."""
    # Select random samples
    indices = np.random.choice(len(X), 10, replace=False)
    X_samples = X[indices]
    y_true = np.argmax(y[indices], axis=1)
    
    # Get predictions
    h1 = X_samples @ model.layers[0].weights.data + model.layers[0].bias.data
    a1 = np.maximum(0, h1)
    h2 = a1 @ model.layers[2].weights.data + model.layers[2].bias.data
    a2 = np.maximum(0, h2)
    logits = a2 @ model.layers[4].weights.data + model.layers[4].bias.data
    probs = softmax(logits)
    predictions = np.argmax(probs, axis=1)
    
    # Create figure
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.ravel()
    
    for idx, (image, true_label, pred_label) in enumerate(zip(X_samples, y_true, predictions)):
        axes[idx].imshow(image.reshape(28, 28), cmap='gray')
        axes[idx].axis('off')
        color = 'green' if true_label == pred_label else 'red'
        axes[idx].set_title(f'True: {true_label}\nPred: {pred_label}', color=color)
    
    plt.suptitle(f'Predictions at Epoch {epoch+1}')
    plt.tight_layout()
    plt.savefig(f'predictions_epoch_{epoch+1}.png')
    plt.close()

def plot_training_curves(train_losses, train_accs, val_accs, val_losses):
    """Plot training and validation curves."""
    plt.figure(figsize=(12, 4))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()

class TrainerWithMetrics(Trainer):
    def evaluate(self, val_loader, compute_loss=False):
        correct = 0
        n_samples = 0
        total_loss = 0
        
        for X, y in val_loader:
            n_samples += X.shape[0]
            
            # Forward pass through all layers
            current_input = X
            for i in range(0, len(self.model.layers), 2):  # Step by 2 to handle weights+activation pairs
                # Linear layer
                h = current_input @ self.model.layers[i].weights.data + self.model.layers[i].bias.data
                # Activation layer
                if i < len(self.model.layers) - 2:  # ReLU for hidden layers
                    current_input = np.maximum(0, h)
                else:  # Softmax for output layer
                    exp_h = np.exp(h - np.max(h, axis=1, keepdims=True))
                    current_input = exp_h / np.sum(exp_h, axis=1, keepdims=True)
            
            probs = current_input  # Final output after all layers
            
            # Ensure predictions are made based on the actual output size
            predictions = np.argmax(probs, axis=1)
            true_labels = np.argmax(y, axis=1)
            correct += np.sum(predictions == true_labels)
            
            if compute_loss:
                # Compute cross-entropy loss
                # Make sure y matches the shape of probs
                if y.shape[1] != probs.shape[1]:
                    raise ValueError(
                        f"Mismatch between model output dimension ({probs.shape[1]}) "
                        f"and target labels dimension ({y.shape[1]}). "
                        "Please ensure your model's output dimension matches the number of classes."
                    )
                eps = 1e-7
                total_loss -= np.sum(y * np.log(probs + eps))
        
        accuracy = correct / n_samples
        if compute_loss:
            loss = total_loss / n_samples
            return loss, accuracy
        return accuracy

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load data
    X_train, y_train_one_hot, _ = load_and_preprocess_data('mnist_train.csv')
    X_test, y_test_one_hot, _ = load_and_preprocess_data('mnist_test.csv')
    
    # Create validation set
    X_val = X_train[-10000:]
    y_val_one_hot = y_train_one_hot[-10000:]
    X_train = X_train[:-10000]
    y_train_one_hot = y_train_one_hot[:-10000]
    
    # Create data loaders
    train_loader = DataLoader(X_train, y_train_one_hot, batch_size=64, shuffle=True)
    val_loader = DataLoader(X_val, y_val_one_hot, batch_size=64, shuffle=False)
    test_loader = DataLoader(X_test, y_test_one_hot, batch_size=64, shuffle=False)
    
    # Initialize model and trainer
    model = MNISTClassifier()
    trainer = TrainerWithMetrics(model, learning_rate=0.01)
    
    # Training loop
    n_epochs = 10
    best_val_acc = 0
    
    # Lists to store metrics
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch+1}/{n_epochs}")
        
        # Train
        train_loss, train_acc = trainer.train_epoch(
            train_loader, 
            desc=f"Epoch {epoch+1}/{n_epochs}"
        )
        
        # Validate
        val_loss, val_acc = trainer.evaluate(val_loader, compute_loss=True)
        
        # Store metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"New best validation accuracy: {val_acc:.4f}")
        
        # Plot sample predictions
        plot_sample_predictions(model, X_val, y_val_one_hot, epoch)
        
        # Plot training curves after each epoch
        plot_training_curves(train_losses, train_accs, val_accs, val_losses)
        
        print('-' * 50)
    
    # Final test evaluation
    test_loss, test_acc = trainer.evaluate(test_loader, compute_loss=True)
    print(f"\nFinal Test Loss: {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    
    # Save final metrics
    np.savez('training_history.npz',
             train_losses=train_losses,
             train_accs=train_accs,
             val_losses=val_losses,
             val_accs=val_accs)

if __name__ == "__main__":
    main()
