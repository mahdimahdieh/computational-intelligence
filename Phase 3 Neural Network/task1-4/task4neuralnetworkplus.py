from neuralnetwork import *
import cupy as cp


# Add Tanh activation function
class Tanh(ActivationFunction):
    def activate(self, x):
        return cp.tanh(x)

    def derivative(self, x):
        return 1 - cp.tanh(x) ** 2


# Optimizer Base Class
class Optimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, layer):
        raise NotImplementedError


# SGD Optimizer
class SGD(Optimizer):
    def update(self, layer):
        layer.weights -= self.learning_rate * layer.grad_weights
        layer.biases -= self.learning_rate * layer.grad_biases


# Momentum Optimizer
class Momentum(Optimizer):
    def __init__(self, learning_rate, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocities = {}

    def update(self, layer):
        # Get or initialize velocity for this layer
        if id(layer) not in self.velocities:
            self.velocities[id(layer)] = {
                'weights': cp.zeros_like(layer.weights),
                'biases': cp.zeros_like(layer.biases)
            }

        v = self.velocities[id(layer)]

        # Update velocities
        v['weights'] = self.momentum * v['weights'] - self.learning_rate * layer.grad_weights
        v['biases'] = self.momentum * v['biases'] - self.learning_rate * layer.grad_biases

        # Update parameters
        layer.weights += v['weights']
        layer.biases += v['biases']


# Adam Optimizer
class Adam(Optimizer):
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-12):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.moments = {}

    def update(self, layer):
        # Get or initialize moments for this layer
        if id(layer) not in self.moments:
            self.moments[id(layer)] = {
                'm_w': cp.zeros_like(layer.weights),
                'v_w': cp.zeros_like(layer.weights),
                'm_b': cp.zeros_like(layer.biases),
                'v_b': cp.zeros_like(layer.biases)
            }

        self.t += 1
        m = self.moments[id(layer)]

        # Update first moment estimates
        m['m_w'] = self.beta1 * m['m_w'] + (1 - self.beta1) * layer.grad_weights
        m['m_b'] = self.beta1 * m['m_b'] + (1 - self.beta1) * layer.grad_biases

        # Update second moment estimates
        m['v_w'] = self.beta2 * m['v_w'] + (1 - self.beta2) * (layer.grad_weights ** 2)
        m['v_b'] = self.beta2 * m['v_b'] + (1 - self.beta2) * (layer.grad_biases ** 2)

        # Bias correction
        m_w_hat = m['m_w'] / (1 - self.beta1 ** self.t)
        v_w_hat = m['v_w'] / (1 - self.beta2 ** self.t)
        m_b_hat = m['m_b'] / (1 - self.beta1 ** self.t)
        v_b_hat = m['v_b'] / (1 - self.beta2 ** self.t)

        # Update parameters
        layer.weights -= self.learning_rate * m_w_hat / (cp.sqrt(v_w_hat) + self.epsilon)
        layer.biases -= self.learning_rate * m_b_hat / (cp.sqrt(v_b_hat) + self.epsilon)



        # Enhanced DenseLayer with weight initialization and gradient storage


# Modified EnhancedDenseLayer
class EnhancedDenseLayer(DenseLayer):
    def __init__(self, input_size, output_size, activation):
        super().__init__(input_size, output_size, activation)
        self.grad_weights = None
        self.grad_biases = None

        # Apply proper weight initialization
        if activation == ReLU:
            # He initialization for ReLU
            scale = cp.sqrt(2.0 / input_size)
        elif activation in [Sigmoid, Tanh]:
            # Xavier/Glorot initialization for Sigmoid/Tanh
            scale = cp.sqrt(1.0 / input_size)
        else:
            scale = 0.01

        self.weights = cp.random.randn(input_size, output_size) * scale
        self.biases = cp.zeros((1, output_size))

    def backward(self, gradient_output, learning_rate=None):
        """Backward pass with gradient storage"""
        activation_gradient = self.activation.derivative(self.z)
        gradient = gradient_output * activation_gradient

        # Store gradients instead of updating parameters
        self.grad_weights = cp.dot(self.x.T, gradient)
        self.grad_biases = cp.sum(gradient, axis=0, keepdims=True)
        gradient_input = cp.dot(gradient, self.weights.T)

        return gradient_input


# Modified EnhancedNeuralNetwork
class EnhancedNeuralNetwork(NeuralNetwork):
    def __init__(self, layers, optimizer):
        # Convert existing DenseLayer to EnhancedDenseLayer if needed
        enhanced_layers = []
        for layer in layers:
            if isinstance(layer, DenseLayer) and not isinstance(layer, EnhancedDenseLayer):
                enhanced_layer = EnhancedDenseLayer(
                    layer.weights.shape[0],
                    layer.weights.shape[1],
                    type(layer.activation)
                )
                # Copy existing weights and biases
                enhanced_layer.weights = layer.weights.copy()
                enhanced_layer.biases = layer.biases.copy()
                enhanced_layers.append(enhanced_layer)
            else:
                enhanced_layers.append(layer)

        super().__init__(enhanced_layers)
        self.optimizer = optimizer

    def backward(self, gradient, learning_rate):
        """Backward pass with optimizer-based updates"""
        # Perform backward pass to compute gradients
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, learning_rate)

        # Update parameters using optimizer
        for layer in self.layers:
            if isinstance(layer, EnhancedDenseLayer):
                self.optimizer.update(layer)

    def train(self, x_train, y_train, x_val, y_val, epochs=100, learning_rate=0.01, batch_size=128,
              classification_task=BinaryClassification):
        """Override train to pass learning_rate to backward"""
        history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}

        for epoch in range(epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            batch_count = 0  # Track number of batches
            # Mini-batch training
            for i in range(0, x_train.shape[0], batch_size):
                x_batch = x_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                # Forward pass
                output = self.forward(x_batch)
                results = classification_task(x_batch, y_batch, output)

                # Backward pass with optimizer update
                self.backward(results.calculate_gradient(), learning_rate)

                # Accumulate metrics
                epoch_loss += results.calculate_loss()
                correct += cp.sum(results.predict() == y_batch)
                total += len(y_batch)

                batch_count += 1

            # Validation
            val_loss, val_acc = self.evaluate(x_val, y_val, classification_task)

            # Calculate epoch averages and convert to float
            avg_loss = float(epoch_loss / batch_count)
            avg_acc = float(correct / total)
            val_loss, val_acc = self.evaluate(x_val, y_val, classification_task)
            val_loss = float(val_loss)
            val_acc = float(val_acc)

            # Record metrics
            history['loss'].append(avg_loss)
            history['acc'].append(avg_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            # Print progress
            print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"Acc: {avg_acc:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f}")

        return history