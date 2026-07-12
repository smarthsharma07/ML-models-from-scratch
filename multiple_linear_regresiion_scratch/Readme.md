# Multiple Linear Regression From Scratch

A NumPy implementation of **Multiple Linear Regression** using the **Normal Equation (Ordinary Least Squares)**.

This project was built to understand the mathematics behind multiple linear regression without relying on machine learning libraries such as **scikit-learn**.

---

## Overview

Multiple Linear Regression models the relationship between one target variable and multiple input features.

The hypothesis function is

\[
\hat{y}=w_1x_1+w_2x_2+\cdots+w_nx_n+b
\]

where

- \(x_i\) are the input features
- \(w_i\) are the learned coefficients
- \(b\) is the intercept (bias)

---

## Mathematical Formulation

The model parameters are obtained by minimizing the Mean Squared Error (MSE).

Using matrix calculus, the solution is given by the **Normal Equation**

\[
\boxed{
w=(X^TX)^{-1}X^Ty
}
\]

where

- **X** = Feature matrix
- **y** = Target vector
- **w** = Weight vector

To incorporate the intercept term, a column of ones is appended to the feature matrix before training.

---

## Features

- Pure NumPy implementation
- Normal Equation (Closed Form Solution)
- Automatic bias handling
- Prediction on unseen samples
- R² Score evaluation
- API similar to scikit-learn

---

## Project Structure

```
multiple_linear_regression/
│
├── multiple_linear_regression.py
├── example.py
├── notebook.ipynb
└── README.md
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/ml-models-from-scratch.git
```

Install dependencies

```bash
pip install numpy
```

---

## Usage

```python
import numpy as np
from multiple_linear_regression import MultipleLinearRegression

X = np.array([
    [1,2],
    [2,1],
    [3,4],
    [4,3],
    [5,5]
])

y = np.array([8,9,16,17,23])

model = MultipleLinearRegression()

model.fit(X, y)

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

predictions = model.predict(X)

print(predictions)

print("R² Score:", model.score(X, y))
```

---

## API

### fit(X, y)

Fits the model using the Normal Equation.

Returns

- Learned coefficients
- Learned intercept

---

### predict(X)

Predicts target values for new input samples.

Returns

- NumPy array containing predictions.

---

### score(X, y)

Computes the coefficient of determination (**R² Score**).

Returns

- Float

---

## Time Complexity

### Training

The Normal Equation requires matrix multiplication and inversion.

- Time Complexity:

\[
O(nm^2+m^3)
\]

where

- **n** = number of samples
- **m** = number of features

### Prediction

\[
O(nm)
\]

---

## Validation

The implementation was validated against **scikit-learn's LinearRegression**.

| Metric | Scratch | Scikit-Learn |
|--------|---------|--------------|
| Intercept | ✅ Matches | ✅ |
| Coefficients | ✅ Matches | ✅ |
| Predictions | ✅ Matches | ✅ |
| R² Score | ✅ Matches | ✅ |

The tiny numerical differences (≈10⁻¹⁴) are due to floating-point precision.

---

## Limitations

Current implementation

- Supports numerical features only.
- Uses the closed-form Normal Equation.
- Intended for educational purposes.

---

## References

- Hands om machine learning with scikit learn
- CampusX
- scikit-learn Documentation
