import numpy as np

class GradientDescent:
    def __init__(self, learning_rate=0.01, epochs=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.tolerance = tolerance

        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for epoch in range(self.epochs):

            predictions = np.dot(X, self.weights) + self.bias

            errors = predictions - y

            dw = (1 / n_samples) * np.dot(X.T, errors)
            db = (1 / n_samples) * np.sum(errors)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            loss = np.mean(errors ** 2)
            self.loss_history.append(loss)

            if np.linalg.norm(dw) < self.tolerance and abs(db) < self.tolerance:
                print(f"Converged after {epoch + 1} epochs")
                break

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def score(self, X, y_test):
        predictions = self.predict(X)

        ss_res = np.sum((y_test - predictions) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)

        r2 = 1 - (ss_res / ss_tot)
        return r2