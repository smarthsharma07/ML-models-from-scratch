from linear_regression import LinearRegressionScratch
import numpy as np

X = np.array([1,2,3,4,5])
y = np.array([2,4,5,4,5])

model = LinearRegressionScratch()
model.fit(X, y)

print(model.weight)
print(model.bias)
print(model.predict([6, 7]))
print(model.score(X, y))