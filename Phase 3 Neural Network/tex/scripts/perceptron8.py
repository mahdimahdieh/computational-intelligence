# perceptron8.py
import numpy as np

patterns = np.array([
    [ 1,  1,  1, -1, -1, -1, -1, -1, -1],  # P1
    [-1, -1, -1,  1,  1,  1, -1, -1, -1],  # P2
    [-1, -1, -1, -1, -1, -1,  1,  1,  1],  # P3
    [ 1, -1, -1,  1, -1, -1,  1, -1, -1],  # P4
    [-1,  1, -1,  1, -1,  1, -1,  1, -1],  # P5
    [-1, -1,  1, -1, -1,  1, -1, -1,  1],  # P6
])

num_classes = patterns.shape[0]
num_features = patterns.shape[1]

targets = -np.ones((num_classes, num_classes))
for i in range(num_classes):
    targets[i, i] = 1

W = np.zeros((num_classes, num_features))   
b = np.zeros(num_classes)                 
eta = 0.1                                   

for i in range(num_classes):
    converged = False
    while not converged:
        converged = True
        for k, x in enumerate(patterns):
            d = targets[i, k]
            z = np.dot(W[i], x) + b[i]
            y = 1 if z >= 0 else -1
            if y != d:
                W[i] += eta * (d - y) * x
                b[i] += eta * (d - y)
                converged = False

for i in range(num_classes):
    print(f"Perceptron {i+1}: w = {W[i]}, b = {b[i]}")
