from polynomial_linear_regression import PolynomialLinearRegression
import numpy as np

regressor = PolynomialLinearRegression(degree=3)
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 4, 9, 16, 25])  
regressor.fit(X, y) 
predict= regressor.predict(np.array([[6], [7]]))
print("Predictions for input [[6], [7]]:", predict)
print("Model score on training data:", regressor.score(X, y))