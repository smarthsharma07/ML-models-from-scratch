# Simple Linear Regression From Scratch

A NumPy implementation of **Simple Linear Regression** using the **Ordinary Least Squares (OLS)** closed-form solution.

This implementation is built for educational purposes to demonstrate how linear regression works internally without using machine learning libraries.

---

## Mathematical Background

Given a dataset

\[
(x_1,y_1),(x_2,y_2),...,(x_n,y_n)
\]

The regression line is

\[
y = wx + b
\]

where

### Weight (Slope)

\[
w=\frac{\sum(x_i-\bar{x})(y_i-\bar{y})}
{\sum(x_i-\bar{x})^2}
\]

### Bias (Intercept)

\[
b=\bar{y}-w\bar{x}
\]

---

## Project Structure

```
Simple Linear Regression/
│
├── linear_regression.py
├── example.py
└── README.md
```

---

## Features

- Closed-form OLS solution
- Pure NumPy implementation
- No external ML libraries
- Predict new values
- R² score implementation

---

## Usage

```python
import numpy as np
from linear_regression import LinearRegressionScratch

X = np.array([1,2,3,4,5])
y = np.array([2,4,5,4,5])

model = LinearRegressionScratch()

model.fit(X, y)

print("Weight:", model.weight)
print("Bias:", model.bias)

predictions = model.predict([6,7])

print(predictions)

print("R²:", model.score(X,y))
```

---

## API

### fit()

Fits the regression line to the training data.

```python
model.fit(X_train, y_train)
```

---

### predict()

Predicts output values for new inputs.

```python
predictions = model.predict(X_test)
```

---

### score()

Returns the coefficient of determination (R²).

```python
score = model.score(X_test, y_test)
```

---

## Complexity

Training

- Time Complexity: **O(n)**
- Space Complexity: **O(1)**

Prediction

- Time Complexity: **O(n)**

---

## Limitations

This implementation currently supports:

- One input feature (Simple Linear Regression)
- Continuous target variable

Future versions will include:

- Multiple Linear Regression
- Gradient Descent implementation
- Regularization techniques
- Polynomial Regression

---

## References

- Hands on machine learning with scikit learn
- CampusX
- scikit-learn Documentation
