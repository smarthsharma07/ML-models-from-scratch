import numpy as np

class StochasticGradientDescentRegressor:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
    def fit(self, X, y):
        self.bias = 0
        self.weights = np.ones(X.shape[1])
        for epoch in range(self.epochs):
            indices = np.random.permutation(X.shape[0])
            for i in indices:
                prediction = np.dot(X[i], self.weights) + self.bias
                error = y[i] - prediction
                dw = -2 * X[i] * error
                db = -2 * error
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    def score(self, X, y):
        y_pred = self.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        return 1 - mse / np.mean((y - np.mean(y)) ** 2)