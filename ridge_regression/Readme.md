# Ridge Regression From Scratch

A complete implementation of **Ridge Regression** built entirely from scratch using **NumPy**.

The objective of this project is to understand how **L2 Regularization** helps reduce overfitting by penalizing large model weights while still learning a linear relationship between input features and the target variable.

This implementation uses **Batch Gradient Descent** for optimization without relying on machine learning libraries such as **scikit-learn**.

---

# Features

- Ridge Regression implemented from scratch
- Supports multiple input features
- Batch Gradient Descent optimization
- L2 (Ridge) Regularization
- Bias term handled separately (not regularized)
- Prediction support
- R² Score evaluation
- Lightweight implementation using only NumPy

---

# Mathematical Formulation

Given a dataset

\[
X \in \mathbb{R}^{m \times n}
\]

where

- \(m\) = number of training examples
- \(n\) = number of features

The prediction function is

\[
\hat{y}=Xw+b
\]

where

- \(w\) = weight vector
- \(b\) = bias

---

## Cost Function

Ridge Regression minimizes the following objective:

\[
J(w,b)=\frac{1}{2m}\sum_{i=1}^{m}(y_i-\hat y_i)^2+\frac{\alpha}{2m}\sum_{j=1}^{n}w_j^2
\]

where

- \(m\) = number of samples
- \(\alpha\) = regularization strength

The second term penalizes large weights, helping reduce overfitting.

---

## Gradient Equations

Weight gradient

\[
\frac{\partial J}{\partial w}
=
\frac{1}{m}
\left(
X^T(\hat y-y)+\alpha w
\right)
\]

Bias gradient

\[
\frac{\partial J}{\partial b}
=
\frac{1}{m}
\sum(\hat y-y)
\]

Parameter update

\[
w=w-\eta\frac{\partial J}{\partial w}
\]

\[
b=b-\eta\frac{\partial J}{\partial b}
\]

where

- \(\eta\) is the learning rate.

---

# Project Structure

```
ridge_regression.py
```

The class provides

- `fit(X, y)`
- `predict(X)`
- `score(X, y)`

---

# Parameters

| Parameter | Description |
|-----------|-------------|
| `learning_rate` | Step size for gradient descent |
| `epochs` | Number of optimization iterations |
| `alpha` | L2 regularization strength |

---

# Methods

## fit(X, y)

Trains the Ridge Regression model using Batch Gradient Descent.

---

## predict(X)

Predicts target values for unseen data.

---

## score(X, y)

Computes the coefficient of determination (**R² Score**).

\[
R^2
=
1-
\frac{\sum(y-\hat y)^2}
{\sum(y-\bar y)^2}
\]

---

# Example

```python
import numpy as np
from ridge_regression import RidgeRegression

X = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6]
])

y = np.array([3, 5, 7, 9, 11])

model = RidgeRegression(
    learning_rate=0.01,
    epochs=1000,
    alpha=1.0
)

model.fit(X, y)

predictions = model.predict(X)

print("Weights:", model.weights)
print("Bias:", model.bias)
print("R² Score:", model.score(X, y))
```

---

# Time Complexity

## Training

For each epoch,

- Prediction: **O(mn)**
- Gradient computation: **O(mn)**

Overall

\[
O(\text{epochs} \times m \times n)
\]

where

- \(m\) = samples
- \(n\) = features

---

## Prediction

\[
O(mn)
\]

---

# Space Complexity

| Component | Complexity |
|------------|------------|
| Weights | O(n) |
| Predictions | O(m) |
| Input Data | O(mn) |

Overall auxiliary space:

\[
O(n)
\]

(excluding the input dataset)

---

# Advantages

- Reduces overfitting
- Handles multicollinearity better than ordinary Linear Regression
- Works well with high-dimensional datasets
- Penalizes large coefficients while retaining all features
- Stable and computationally efficient

---

# Limitations

- Does not perform feature selection (weights shrink but rarely become exactly zero)
- Assumes a linear relationship between features and target
- Sensitive to feature scaling (standardization is recommended)

---

# Future Improvements

- Mini-Batch Gradient Descent
- Stochastic Gradient Descent
- Early Stopping
- Learning Rate Scheduling
- Cost History Tracking
- Model Serialization
- Cross-validation for selecting the optimal regularization parameter

---

# Dependencies

- Python 3.x
- NumPy

Install NumPy using

```bash
pip install numpy
```

---

# License

This project is open-source and intended for educational purposes to understand the implementation of Ridge Regression from first principles.
