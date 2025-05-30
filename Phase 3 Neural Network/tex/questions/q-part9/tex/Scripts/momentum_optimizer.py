def update_params(self, grads_w, grads_b):
    for i in range(len(self.weights)):
        self.vel_w[i] = self.momentum * self.vel_w[i] - self.lr * grads_w[i]
        self.vel_b[i] = self.momentum * self.vel_b[i] - self.lr * grads_b[i]
        self.weights[i] += self.vel_w[i]
        self.biases[i] += self.vel_b[i]