import numpy as np


class MultipleLinearRegression:
    """
    Multiple Linear Regression implemented from scratch
    using the Normal Equation(OLS).
    """

    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def _add_bias(self, X):
        """Adds a column of ones for the bias term."""
        return np.insert(X, 0, 1, axis=1)

    def fit(self, X_train, y_train):
        """
        We Train the model using the Normal Equation(OLS).

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
        y_train : array-like of shape (n_samples,)
        """

        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)

        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                "X_train and y_train must have the same number of samples."
            )

        X_train = self._add_bias(X_train)

        weights = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train

        self.intercept_ = weights[0]
        self.coef_ = weights[1:]

        return self

    def predict(self, X_test):
        """
        Predict target values.
        """

        if self.coef_ is None:
            raise ValueError("Model has not been fitted yet.")

        X_test = np.asarray(X_test, dtype=float)
        X_test = self._add_bias(X_test)

        weights = np.insert(self.coef_, 0, self.intercept_)

        return X_test @ weights

    def score(self, X_test, y_test):
        """
        Returns the R² score.
        """

        y_test = np.asarray(y_test, dtype=float)

        y_pred = self.predict(X_test)

        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)

        return 1 - (ss_res / ss_tot)