import cupy as cp

# Abstract Activation Function
class ActivationFunction:
    def activate(self, x):
        raise NotImplementedError

    def derivative(self, x):
        raise NotImplementedError

# Sigmoid Activation Function
class Sigmoid(ActivationFunction):
    def activate(self, x):
        return 1 / (1 + cp.exp(-x))

    def derivative(self, x):
        """Derivative of sigmoid function."""
        s = self.activate(x)
        return s * (1 - s)  # Derivative of sigmoid

# Softmax Activation Function
class Softmax(ActivationFunction):
    def activate(self, x):
        e_x = cp.exp(x - cp.max(x, axis=1, keepdims=True))  # Numerically stable softmax
        return e_x / cp.sum(e_x, axis=1, keepdims=True)  # Normalize to probabilities

    def derivative(self, x):
        """
        Derivative of softmax function is very complex:
        https://medium.com/data-science/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
        Since it used only in the output, we leave it as it is.
        """
        return 1

# ReLU Activation Function
class ReLU(ActivationFunction):
    def activate(self, x):
        return cp.maximum(0, x)  # Element-wise ReLU

    def derivative(self, x):
        return cp.where(x > 0, cp.ndarray(1.0), cp.ndarray(0.0))  # 1 if x > 0, else 0


# Abstract Loss Function + Prediction + Gradient
class ClassificationTask:
    def __init__(self,x_batch, y_batch, output):
        self.x_batch = x_batch
        self.y_batch = y_batch
        self.output = output

    def calculate_loss(self):
        raise NotImplementedError
    def predict(self):
        raise NotImplementedError
    def calculate_gradient(self):
        raise NotImplementedError

class BinaryClassification(ClassificationTask):
    def __init__(self, x_batch, y_batch, output):
        super().__init__(x_batch, y_batch, output)

    # Binary Cross Entropy
    def calculate_loss(self, epsilon=1e-12):
        # clip: ``maximum(minimum(a, a_max), a_min)`` to keep the prediction in range to avoid log(0)
        output = cp.clip(self.output, epsilon, 1 - epsilon)
        return -cp.mean(output * cp.log(output) + (1 - output) * cp.log(1 - output))

    def predict(self):
        return (self.output > 0.5).astype(int)

    def calculate_gradient(self):
        return (self.output - self.y_batch.reshape(-1, 1)) / self.x_batch.shape[0]

class CategoricalClassification(ClassificationTask):
    def __init__(self, x_batch, y_batch, output):
        super().__init__(x_batch, y_batch, output)
        self.y_onehot = cp.eye(10)[self.y_batch.get().astype(int)]  # One-hot encode labels

    # Categorical Cross-Entrpy Loss Function
    def calculate_loss(self, epsilon=1e-12):
        # clip: `maximum(minimum(a, a_max), a_min)` to keep the prediction in range to avoid log(0)
        self.output = cp.clip(self.output, epsilon, 1)
        return -cp.mean(cp.sum(self.y_onehot * cp.log(self.output), axis=1))  # Categorical cross entropy formula


    def predict(self):
        return cp.argmax(self.output, axis=1)

    def calculate_gradient(self):
        return (self.output - self.y_onehot) / self.x_batch.shape[0]  # Multiclass gradient

class DenseLayer:
    """A fully connected neural network layer."""
    def __init__(self, input_size, output_size, activation):
        self.weights = cp.random.randn(input_size, output_size) * 0.01
        self.biases = cp.zeros((1, output_size))
        self.activation = activation()
        self.x = None  # Will store layer inputs
        self.z = None  # Will store pre-activation outputs
        self.output = None  # Will store post-activation outputs

    def forward(self, x):
        """Forward pass through the layer."""
        self.x = x
        self.z = cp.dot(x, self.weights) + self.biases  # Linear transformation

        # Apply activation function
        self.output = self.activation.activate(self.z)

        return self.output

    def backward(self, gradient_output, learning_rate):
        """Backward pass (backpropagation) through the layer.

        Args:
            gradient_output: Gradient from next layer
            learning_rate: Learning rate for weight updates

        Returns:
            Gradient to pass to previous layer
        """
        activation_gradient = self.activation.derivative(self.z)

        gradient = gradient_output * activation_gradient  # Chain rule
        gradient_weights = cp.dot(self.x.T, gradient)  # Gradient for weights
        gradient_biases = cp.sum(gradient, axis=0, keepdims=True)  # Gradient for biases
        gradient_input = cp.dot(gradient, self.weights.T)  # Gradient for input (to pass to previous layer)

        # Update parameters using gradient descent
        self.weights -= learning_rate * gradient_weights
        self.biases -= learning_rate * gradient_biases

        return gradient_input


class NeuralNetwork:
    """Implementation of a multi-layer neural network."""

    def __init__(self, layers):
        """Initialize network with list of layers."""
        self.layers = layers

    def forward(self, x):
        """Forward pass through entire network."""
        for layer in self.layers:
            x = layer.forward(x)  # Pass input through each layer
        return x

    def backward(self, gradient, learning_rate):
        """Backward pass through entire network."""
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, learning_rate)  # Backpropagate through each layer

    def train(self, x_train, y_train, x_val, y_val, epochs=100, learning_rate=0.01, batch_size=128, classification_task=BinaryClassification):
        """Train the neural network.

        Args:
            x_train: Training data
            y_train: Training labels
            x_val: Validation data
            y_val: Validation labels
            epochs: Number of training epochs
            learning_rate: Learning rate for gradient descent
            batch_size: Size of mini-batches
            classification_task: binary or multiclass classification

        Returns:
            Dictionary with training history
        """
        history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}

        for epoch in range(epochs):
            epoch_loss = 0
            correct = 0
            total = 0

            # Mini-batch training
            for i in range(0, x_train.shape[0], batch_size):
                x_batch = x_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                # Forward pass
                output = self.forward(x_batch)

                classification_results = classification_task(x_batch, y_batch, output)

                self.backward(classification_results.calculate_gradient(), learning_rate)

                # Accumulate metrics
                epoch_loss += classification_results.calculate_loss()
                correct += cp.sum(cp.ndarray(classification_results.predict() == y_batch))
                total += len(y_batch)
            # Validation
            val_loss, val_acc = self.evaluate(x_val, y_val, classification_task)

            # Record metrics
            history['loss'].append(epoch_loss / (i // batch_size + 1))
            history['acc'].append(correct / total)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            # Print progress
            print(f"Epoch {epoch + 1}/{epochs} | Loss: {history['loss'][-1]:.4f} | "
                  f"Acc: {history['acc'][-1]:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f}")

        return history

    def evaluate(self, x, y, classification_task):
        """Evaluate model on given data.

        Args:
            x: Input data
            y: True labels
            classification_task: binary or multiclass classification

        Returns:
            Tuple of (loss, accuracy)
        """
        output = self.forward(x)

        classification_results = classification_task(x, y, output)

        acc = cp.mean(cp.ndarray(classification_results.predict() == y))
        return classification_results.calculate_loss(), acc

