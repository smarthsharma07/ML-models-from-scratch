from RidgeRegression import RidgeRegression
import numpy as np

regressor = RidgeRegression(learning_rate=0.01, epochs=1000, alpha=1.0)
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 4, 9, 16, 25])
regressor.fit(X, y)
predict = regressor.predict(np.array([[6], [7]]))
print("Predictions for input [[6], [7]]:", predict)
print("Model score on training data:", regressor.score(X, y))