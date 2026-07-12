import numpy as np

from multiple_LR import MultipleLinearRegression

X = np.array([
    [1, 2],
    [2, 1],
    [3, 4],
    [4, 3],
    [5, 5]
])

y = np.array([8, 9, 16, 17, 23])

model = MultipleLinearRegression()

model.fit(X, y)

print("Intercept.scratch:", model.intercept_)
print("Coefficients.scratch:", model.coef_)

predictions = model.predict(X)

print(predictions)

print("R² Score:", model.score(X, y))

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X, y)

print("Intercept.sklearn:", lr.intercept_)
print("Coefficients.sklearn:", lr.coef_)
print("R² Score.sklearn:", lr.score(X, y))