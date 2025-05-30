import cupy as cp
from overrides import overrides


# Abstract Activation function
class ActivationFunction:
    def activate(self, x):
        raise NotImplementedError

    def derivative(self, x):
        raise NotImplementedError

# Sigmoid Activation function
class Sigmoid(ActivationFunction):
    @overrides
    def activate(self, x):
        return 1 / (1 + cp.exp(-x))

    @overrides
    def derivative(self, x):
        """Derivative of sigmoid function."""
        s = self.activate(x)
        return s * (1 - s)  # Derivative of sigmoid


# Loss Functions
def binary_cross_entropy(y_true, y_pred, epsilon=1e-12):
    # clip: ``maximum(minimum(a, a_max), a_min)`` to keep the prediction in range to avoid log(0)
    y_pred = cp.clip(y_pred, epsilon, 1 - epsilon)
    return -cp.mean(y_true * cp.log(y_pred) + (1 - y_true) * cp.log(1 - y_pred))


class DenseLayer:
    """A fully connected neural network layer."""
    def __init__(self, input_size, output_size, activation=Sigmoid):
        self.weights = cp.zeros(input_size, output_size)
        self.biases = cp.zeros((1, output_size))
        self.activation = activation
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
        activation_gradient = self.activation.derivative(self.x)

        gradient = gradient_output * activation_gradient  # Chain rule
        gradient_weights = cp.dot(self.x.T, gradient)  # Gradient for weights
        gradient_biases = cp.sum(gradient, axis=0, keepdims=True)  # Gradient for biases
        gradient_input = cp.dot(gradient, self.weights.T)  # Gradient for input (to pass to previous layer)

        # Update parameters using gradient descent
        self.weights -= learning_rate * gradient_weights
        self.biases -= learning_rate * gradient_biases

        return gradient_input




