import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import cupy as cp
import numpy as np

def plot_history(history):
    """Plot training and validation loss and accuracy over epochs.

    Args:
        history: Dictionary containing training history metrics
    """
    # Extract metrics from history
    loss = [float(val) for val in history['loss']]
    val_loss = [float(val) for val in history['val_loss']]
    acc = [float(val) for val in history['acc']]
    val_acc = [float(val) for val in history['val_acc']]

    epochs = range(1, len(loss) + 1)

    # Create figure with two subplots
    plt.figure(figsize=(14, 5))

    # Plot loss on first subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Plot accuracy on second subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix_and_f1(y_true, y_pred, classes=None, title='Confusion Matrix'):
    """Plot confusion matrix and print F1 scores.

    Args:
        y_true: True labels (CuPy or NumPy array)
        y_pred: Predicted labels (CuPy or NumPy array)
        classes: List of class names
        title: Title for the plot
    """
    # Convert to NumPy arrays if using CuPy
    if isinstance(y_true, cp.ndarray):
        y_true = y_true.get()
    if isinstance(y_pred, cp.ndarray):
        y_pred = y_pred.get()

    # Flatten arrays if needed
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Calculate classification report
    report = classification_report(y_true, y_pred, target_names=classes, digits=4)

    # Print F1 scores
    print("Classification Report:")
    print(report)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    if classes is not None:
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()