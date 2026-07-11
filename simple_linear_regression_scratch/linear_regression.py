import numpy as np


class LinearRegressionScratch:
    """
    Simple Linear Regression implemented from scratch.
    """

    def __init__(self):
        self.weight = None
        self.bias = None

    def fit(self, X_train, y_train):

        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)

        if len(X_train) != len(y_train):
            raise ValueError("Input arrays must have the same length.")

        x_mean = np.mean(X_train)
        y_mean = np.mean(y_train)

        numerator = np.sum((X_train - x_mean) * (y_train - y_mean))
        denominator = np.sum((X_train - x_mean) ** 2)

        if denominator == 0:
            raise ValueError("Variance of X cannot be zero.")

        self.weight = numerator / denominator
        self.bias = y_mean - self.weight * x_mean

    def predict(self, X_test):
        X_test = np.asarray(X_test)
        return self.weight * X_test + self.bias

    def score(self, X, y):
        y_pred = self.predict(X)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        return 1 - ss_res / ss_tot