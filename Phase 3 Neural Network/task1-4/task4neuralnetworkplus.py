from neuralnetwork import *
import cupy as cp


# Tangent Hyperbolic Activation Function
class Tanh(ActivationFunction):
    def activate(self, x):
        return cp.tanh(x)

    def derivative(self, x):
        return 1 - cp.tanh(x)**2


# Abstract Weights Initializer
class WeightsInitializer:
    def generate_weights(self, input_size, output_size):
        raise NotImplementedError

class HeInitialization(WeightsInitializer):
    def generate_weights(self, input_size, output_size):
        return cp.random.randn(input_size, output_size) * cp.sqrt(2.0 / input_size)

class XavierInitialization(WeightsInitializer):
    def generate_weights(self, input_size, output_size):
        return cp.random.randn(input_size, output_size) * cp.sqrt(1.0 / input_size)


class DenseLayerPlus(DenseLayer):
    """A fully connected neural network layer."""
    def __init__(self, input_size, output_size, activation, weights_initializer = None):
        super().__init__(input_size, output_size, activation)
        if weights_initializer is None:
            weights_initializer = self._default_weights_initialization
        self.weights = weights_initializer(input_size, output_size)
        self.grad_weights = None
        self.grad_biases = None


    @staticmethod
    def _default_weights_initialization(input_size, output_size):
        return cp.random.randn(input_size, output_size) * 0.01

    def backward(self, gradient_output, learning_rate):
        activation_gradient = self.activation.derivative(self.z)
        gradient = gradient_output * activation_gradient
        self.grad_weights = cp.dot(self.x.T, gradient)
        self.grad_biases = cp.sum(gradient, axis=0, keepdims=True)
        gradient_input = cp.dot(gradient, self.weights.T)
        return gradient_input

    def update(self, learning_rate, optimizer=None, **optimizer_params):
        if optimizer == 'momentum':
            beta = optimizer_params.get('beta', 0.9)
            if not hasattr(self, 'velocity_w'):
                self.velocity_w = cp.zeros_like(self.weights)
                self.velocity_b = cp.zeros_like(self.biases)

            self.velocity_w = beta * self.velocity_w + learning_rate * self.grad_weights
            self.velocity_b = beta * self.velocity_b + learning_rate * self.grad_biases
            self.weights -= self.velocity_w
            self.biases -= self.velocity_b

        elif optimizer == 'adam':
            # Adam implementation would go here
            pass
        else:  # SGD
            self.weights -= learning_rate * self.grad_weights
            self.biases -= learning_rate * self.grad_biases
