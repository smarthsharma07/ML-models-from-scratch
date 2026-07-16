import numpy as np
from itertools import combinations_with_replacement

class PolynomialLinearRegression:
    def __init__(self, degree=2):
        if degree < 1:
            raise ValueError("degree must be >= 1")
        self.degree = degree
        self.weights = None
    def _polynomial_features(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_samples, n_features = X.shape
        features = [np.ones((n_samples, 1))]
        for deg in range(1, self.degree + 1):
            for comb in combinations_with_replacement(range(n_features), deg):
                feature = np.prod(X[:, comb], axis=1).reshape(-1, 1)
                features.append(feature)
        return np.hstack(features)
    def fit(self, X, y):
        X_poly = self._polynomial_features(X)
        self.weights = np.linalg.pinv(X_poly.T @ X_poly) @ X_poly.T @ y
    def predict(self, X):
        X_poly = self._polynomial_features(X)
        return X_poly @ self.weights
    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot