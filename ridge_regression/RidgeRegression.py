import numpy as np

class RidgeRegression:
    def __init__(self, learning_rate=0.01, epochs=1000, alpha=1.0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.alpha = alpha
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * (np.dot(X.T, (y_predicted - y)) + self.alpha * self.weights)
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def score(self, X, y):
        y_predicted = self.predict(X)
        ss_res = np.sum((y - y_predicted) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot