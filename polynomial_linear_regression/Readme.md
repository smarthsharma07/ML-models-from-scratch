# Polynomial Linear Regression From Scratch

A NumPy implementation of **Polynomial Linear Regression** built completely from scratch using the **Normal Equation**. This implementation expands the original feature space into polynomial features (including interaction terms) and then fits a linear regression model on the transformed data.

The goal of this implementation is to understand how polynomial regression works mathematically instead of relying on libraries such as scikit-learn.

---

## Features

* Polynomial feature expansion
* Supports multiple input features
* Generates interaction terms
* Uses the Moore-Penrose Pseudoinverse for numerical stability
* Implements prediction and R² score
* Built entirely using NumPy

---

## Mathematical Idea

Polynomial Regression is simply **Linear Regression on transformed features**.

Instead of fitting
$$
[
y = w_0 + w_1x
]
$$
we first transform the input into polynomial features.

For example, if
$$
[
x = [x]
]
$$
and degree = 3,

the transformed feature vector becomes
$$
[
[1,;x,;x^2,;x^3]
]
$$
Linear Regression is then applied to this transformed dataset.

For multiple variables,
$$
[
(x_1,x_2)
]
$$
Degree 2 becomes
$$
[
[1,;
x_1,;
x_2,;
x_1^2,;
x_1x_2,;
x_2^2]
]
$$
The interaction terms allow the model to learn nonlinear relationships between multiple variables.

---

## Project Structure

```text
PolynomialLinearRegression/
│
├── polynomial_linear_regression.py
├── README.md
└── derivation.md
```

---

## Example

```python
import numpy as np
from polynomial_linear_regression import PolynomialLinearRegression

X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5]
])

y = np.array([2,5,10,17,26])

model = PolynomialLinearRegression(degree=2)

model.fit(X,y)

predictions = model.predict(X)

print(predictions)
print(model.score(X,y))
```

---

## API

### Constructor

```python
PolynomialLinearRegression(degree=2)
```

Parameters

* `degree` : Maximum polynomial degree.

---

### fit()

```python
fit(X, y)
```

Fits the polynomial regression model using the Normal Equation.

---

### predict()

```python
predict(X)
```

Returns predictions for the input samples.

---

### score()

```python
score(X, y)
```

Returns the coefficient of determination (R² score).

---

## Time Complexity

### Polynomial Feature Generation

Depends on

* Number of samples (n)
* Number of original features (m)
* Polynomial degree (d)

The number of generated polynomial features is
$$
[
\binom{m+d}{d}
]
$$
Feature generation complexity is approximately
$$
[
O\left(n\binom{m+d}{d}\right)
]
$$
---

### Model Training

The Normal Equation computes
$$
[
(X^TX)^{-1}X^Ty
]
$$
or equivalently uses the pseudoinverse.

If there are (p) polynomial features,

training complexity is approximately
$$
[
O(np^2+p^3)
]
$$
---

### Prediction

Prediction performs one matrix multiplication.

Complexity
$$
[
O(np)
]
$$
---

## Advantages

* Can model nonlinear relationships
* Still produces a linear optimization problem
* Closed-form solution using the Normal Equation
* Easy to understand mathematically

---

## Limitations

* Number of features grows rapidly with degree
* Can easily overfit on small datasets
* Normal Equation becomes expensive for very large feature spaces
* Sensitive to feature scaling for high-degree polynomials

---


## Dependencies

* NumPy

Install

```bash
pip install numpy
```

---

## License

This project is intended for educational purposes to demonstrate the mathematical foundations and implementation details of Polynomial Linear Regression from scratch.
