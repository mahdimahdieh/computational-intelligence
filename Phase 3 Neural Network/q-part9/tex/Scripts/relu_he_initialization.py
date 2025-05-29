def init_weights(self):
        self.weights = []
        self.biases = []
        for i in range(len(self.layer_dims) - 1):
            W = np.random.randn(self.layer_dims[i], self.layer_dims[i + 1]) * np.sqrt(2. / self.layer_dims[i])
            b = np.zeros((1, self.layer_dims[i + 1]))
            self.weights.append(W)
            self.biases.append(b)