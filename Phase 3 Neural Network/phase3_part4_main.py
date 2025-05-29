import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
import torch


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    sig = sigmoid(x)
    return sig * (1 - sig)


def relu(x):
    return np.maximum(0, x)


def relu_deriv(x):
    return (x > 0).astype(float)


ACTIVATIONS = {
    'sigmoid': (sigmoid, sigmoid_deriv),
    'relu': (relu, relu_deriv)
}


class NeuralNetwork:
    def __init__(self, layer_dims, activations, optimizer_cfg=None):
        self.layer_dims = layer_dims
        self.activations = activations
        self.lr = optimizer_cfg.get('lr', 0.01)
        self.momentum = optimizer_cfg.get('momentum', 0)
        self.init_weights()
        self.init_velocity()
        self.history = {'train_loss': [], 'train_acc': []}

    def init_weights(self):
        self.weights = []
        self.biases = []
        for i in range(len(self.layer_dims) - 1):
            W = np.random.randn(self.layer_dims[i], self.layer_dims[i + 1]) * np.sqrt(2. / self.layer_dims[i])
            b = np.zeros((1, self.layer_dims[i + 1]))
            self.weights.append(W)
            self.biases.append(b)

    def init_velocity(self):
        self.vel_w = [np.zeros_like(W) for W in self.weights]
        self.vel_b = [np.zeros_like(b) for b in self.biases]

    def forward(self, X):
        A = X
        self.zs = []
        self.activs = [X]
        for i in range(len(self.weights)):
            Z = A @ self.weights[i] + self.biases[i]
            self.zs.append(Z)
            if i == len(self.weights) - 1:
                A = sigmoid(Z)
            else:
                act_fn, _ = ACTIVATIONS[self.activations[i]]
                A = act_fn(Z)
            self.activs.append(A)
        return A, self.zs

    def backward(self, X, y, output):
        m = X.shape[0]
        grads_w = [0] * len(self.weights)
        grads_b = [0] * len(self.biases)

        delta = output - y.reshape(-1, 1)

        for i in reversed(range(len(self.weights))):
            A_prev = self.activs[i]
            grads_w[i] = A_prev.T @ delta / m
            grads_b[i] = np.sum(delta, axis=0, keepdims=True) / m

            if i != 0:
                _, d_act = ACTIVATIONS[self.activations[i - 1]]
                delta = (delta @ self.weights[i].T) * d_act(self.zs[i - 1])

        return grads_w, grads_b

    def update_params(self, grads_w, grads_b):
        for i in range(len(self.weights)):
            self.vel_w[i] = self.momentum * self.vel_w[i] - self.lr * grads_w[i]
            self.vel_b[i] = self.momentum * self.vel_b[i] - self.lr * grads_b[i]
            self.weights[i] += self.vel_w[i]
            self.biases[i] += self.vel_b[i]

    def train(self, train_loader, epochs=10, print_freq=1):
        for epoch in range(epochs):
            all_loss = []
            all_acc = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.numpy()
                y_batch = y_batch.numpy()
                output, _ = self.forward(x_batch)
                loss = np.mean((output - y_batch.reshape(-1, 1)) ** 2)
                preds = (output >= 0.5).astype(int).flatten()
                acc = np.mean(preds == y_batch)

                grads_w, grads_b = self.backward(x_batch, y_batch, output)
                self.update_params(grads_w, grads_b)

                all_loss.append(loss)
                all_acc.append(acc)

            avg_loss = np.mean(all_loss)
            avg_acc = np.mean(all_acc)
            self.history['train_loss'].append(avg_loss)
            self.history['train_acc'].append(avg_acc)

            if epoch % print_freq == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {avg_acc:.4f}")


def load_data_no_augmentation(positive_class=0, data_dir='./data'):
    transform = T.ToTensor()

    train_set = CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_set = CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    x_train = train_set.data.astype(np.float32) / 255.0
    y_train = np.array(train_set.targets).flatten()
    y_train_bin = (y_train == positive_class).astype(int)

    x_test = test_set.data.astype(np.float32) / 255.0
    y_test = np.array(test_set.targets).flatten()
    y_test_bin = (y_test == positive_class).astype(int)

    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    return x_train, y_train_bin, x_test, y_test_bin


class SimpleLoader:
    def __init__(self, X, y, batch_size=128):
        self.X = X
        self.y = y
        self.batch_size = batch_size

    def __iter__(self):
        indices = np.random.permutation(len(self.X))
        for start in range(0, len(self.X), self.batch_size):
            end = start + self.batch_size
            batch_idx = indices[start:end]
            yield torch.tensor(self.X[batch_idx]), torch.tensor(self.y[batch_idx])


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data_no_augmentation()

    configs = [
        {'name': 'GD-Sigmoid', 'activations': ['sigmoid', 'sigmoid'], 'opt': {'lr': 0.01, 'momentum': 0}},
        {'name': 'Momentum-ReLU', 'activations': ['relu', 'sigmoid'], 'opt': {'lr': 0.01, 'momentum': 0.9}},
    ]

    trained_models = {}
    results = {}

    for cfg in configs:
        print(f"\n== Training {cfg['name']} Without Augmentation ==")
        model = NeuralNetwork(layer_dims=[3072, 64, 1], activations=cfg['activations'], optimizer_cfg=cfg['opt'])
        train_loader = SimpleLoader(x_train, y_train)
        model.train(train_loader, epochs=10, print_freq=2)
        results[cfg['name']] = model.history
        trained_models[cfg['name']] = model

    # Plot Loss
    plt.figure()
    for name, hist in results.items():
        plt.plot(hist['train_loss'], label=name)
    plt.title('Training Loss Comparison (No Augmentation)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot Accuracy
    plt.figure()
    for name, hist in results.items():
        plt.plot(hist['train_acc'], label=name)
    plt.title('Training Accuracy Comparison (No Augmentation)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.show()

    # Evaluation
    print("\n-- Test Set Evaluation --")
    for name, model in trained_models.items():
        yh, _ = model.forward(x_test)
        yp = (yh >= 0.5).astype(int).flatten()
        print(f"\n{name}:\n", confusion_matrix(y_test, yp))
        print(classification_report(y_test, yp, digits=4))

# == Training GD-Sigmoid Without Augmentation ==
# Epoch 1/10 - Loss: 0.0931 - Acc: 0.8955
# Epoch 3/10 - Loss: 0.0809 - Acc: 0.9000
# Epoch 5/10 - Loss: 0.0773 - Acc: 0.9002
# Epoch 7/10 - Loss: 0.0756 - Acc: 0.9011
# Epoch 9/10 - Loss: 0.0746 - Acc: 0.9025
#
# == Training Momentum-ReLU Without Augmentation ==
# Epoch 1/10 - Loss: 0.0812 - Acc: 0.9012
# Epoch 3/10 - Loss: 0.0713 - Acc: 0.9092
# Epoch 5/10 - Loss: 0.0673 - Acc: 0.9136
# Epoch 7/10 - Loss: 0.0652 - Acc: 0.9156
# Epoch 9/10 - Loss: 0.0637 - Acc: 0.9177
#
# -- Test Set Evaluation --
#
# GD-Sigmoid:
#  [[8980   20]
#  [ 942   58]]
#               precision    recall  f1-score   support
#
#            0     0.9051    0.9978    0.9492      9000
#            1     0.7436    0.0580    0.1076      1000
#
#     accuracy                         0.9038     10000
#    macro avg     0.8243    0.5279    0.5284     10000
# weighted avg     0.8889    0.9038    0.8650     10000
#
#
# Momentum-ReLU:
#  [[8826  174]
#  [ 639  361]]
#               precision    recall  f1-score   support
#
#            0     0.9325    0.9807    0.9560      9000
#            1     0.6748    0.3610    0.4704      1000
#
#     accuracy                         0.9187     10000
#    macro avg     0.8036    0.6708    0.7132     10000
# weighted avg     0.9067    0.9187    0.9074     10000
