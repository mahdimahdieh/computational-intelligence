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