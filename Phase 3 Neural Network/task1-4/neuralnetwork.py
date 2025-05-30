import cupy as cp

# Activation Functions
def sigmoid(x):
    return 1 / (1 + cp.exp(-x))

# Loss Functions
def binary_cross_entropy(y_true, y_pred, epsilon=1e-12):
    y_pred = cp.clip(y_pred, epsilon, 1 - epsilon)
    return -cp.mean(y_true * cp.log(y_pred) + (1 - y_true) * cp.log(1 - y_pred))

