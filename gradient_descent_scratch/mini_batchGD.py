import numpy as np

class MiniBatchGradientDescentRegressor:
    def __init__(self, learning_rate=0.01, epochs=100, batch_size=32):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        self.bias = 0
        self.weights = np.zeros(X.shape[1])
        for epoch in range(self.epochs):
            indices = np.random.permutation(X.shape[0])
            for start in range(0, X.shape[0], self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                prediction = np.dot(X_batch, self.weights) + self.bias
                error = y_batch - prediction
                dw = (-2 / X_batch.shape[0]) * np.dot(X_batch.T, error)
                db = -2 * np.mean(error)
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def score(self, X, y):
        y_pred = self.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        return 1 - mse / np.mean((y - np.mean(y)) ** 2)
        