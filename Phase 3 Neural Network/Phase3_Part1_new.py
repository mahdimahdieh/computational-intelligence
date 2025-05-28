# Cell 1:
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from sklearn.metrics import confusion_matrix, f1_score, classification_report

np.random.seed(42)


# Cell 2:
def load_and_preprocess(positive_class=0):
    # Load dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    # Binary label: airplane=1, other=0
    y_train_bin = (y_train == positive_class).astype(int)
    y_test_bin = (y_test == positive_class).astype(int)

    # Normalize to [0,1]
    x_train = x_train.astype(np.float32) / 255.
    x_test = x_test.astype(np.float32) / 255.

    # Flattening to 3072-dimensional vector
    m_train = x_train.shape[0]
    m_test = x_test.shape[0]
    X_train = x_train.reshape(m_train, -1)  # (m_train, 3072)
    X_test = x_test.reshape(m_test, -1)  # (m_test,  3072)

    return X_train, y_train_bin, X_test, y_test_bin


# Cell 3:
def _compute_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


class LogisticRegression:
    def __init__(self, input_dim, lr=0.01, n_iters=2000, print_cost=False):
        self.W = np.random.randn(input_dim) * 0.01  # **Small** random weights
        self.b = 0.0
        self.lr = lr
        self.n_iters = n_iters
        self.print_cost = print_cost
        self.losses = []

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        m = X.shape[0]

        for i in range(self.n_iters):
            # forward
            z = X.dot(self.W) + self.b  # (m,)
            a = self.sigmoid(z)  # (m,)

            # loss
            loss = _compute_loss(y, a)
            self.losses.append(loss)

            # gradients
            dz = a - y  # (m,)
            dw = (1 / m) * X.T.dot(dz)  # (features,)
            db = (1 / m) * np.sum(dz)  # scalar

            # update
            self.W -= self.lr * dw
            self.b -= self.lr * db

            if self.print_cost and i % 100 == 0:
                print(f"Iter {i:4d} : loss: {loss:.4f}")

    def predict_proba(self, X):
        z = X.dot(self.W) + self.b
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


X_train, y_train, X_test, y_test = load_and_preprocess(positive_class=0)


# Cell 4:

# Create Model
model = LogisticRegression(
    input_dim=X_train.shape[1],
    lr=0.01,
    n_iters=10000,
    print_cost=True
)

# train and fit
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

def evaluate(y_test, y_pred):
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("F1 Score:\n", f1_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))

evaluate(y_test, y_pred)

# Visualization of Error
def visualize(model_data):
    plt.plot(model_data.losses)
    plt.title("Training Loss over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.show()

visualize(model)
