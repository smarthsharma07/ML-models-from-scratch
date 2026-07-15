# Gradient Descent Algorithms for Linear Regression (From Scratch)

A NumPy implementation of **Linear Regression** trained using multiple **Gradient Descent optimization algorithms**, built completely from scratch without relying on machine learning libraries such as **scikit-learn**.

The goal of this project is to understand how optimization algorithms work internally by implementing them from first principles.

---

## Implemented Algorithms

- ✅ Batch Gradient Descent
- ✅ Stochastic Gradient Descent (SGD)
- ✅ Mini-Batch Gradient Descent

---

## Features

- Pure NumPy implementation
- Learnable weights and bias
- Mean Squared Error (MSE) loss
- Configurable learning rate
- Configurable number of epochs
- Configurable batch size (Mini-Batch GD)
- Dataset shuffling for SGD and Mini-Batch GD
- Prediction on unseen data
- R² score evaluation

---

## Mathematical Model

For a dataset with one or more features,

\[
\hat{y}=Xw+b
\]

where

- **X** → Feature matrix
- **w** → Weight vector
- **b** → Bias
- **ŷ** → Predicted value

The objective is to minimize the **Mean Squared Error (MSE)**

\[
MSE=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2
\]

using gradient-based optimization.

---

## Optimization Algorithms

### Batch Gradient Descent

Uses the **entire dataset** to compute the gradients before every parameter update.

**Advantages**
- Stable convergence
- Exact gradient

**Disadvantages**
- Slow for very large datasets

---

### Stochastic Gradient Descent (SGD)

Updates the parameters after **every training example**.

**Advantages**
- Faster updates
- Escapes shallow local minima more easily
- Suitable for online learning

**Disadvantages**
- Noisy convergence

---

### Mini-Batch Gradient Descent

Updates the parameters using a **small batch of samples** instead of the entire dataset.

This combines the stability of Batch GD with the speed of SGD and is the optimizer most commonly used in modern machine learning and deep learning.

---

## Project Structure

```
gradient_descent_scratch/
│
├── gradient_descent.py      # Batch Gradient Descent
├── Stochastic_GD.py         # Stochastic Gradient Descent
├── mini_batchGD.py          # Mini-Batch Gradient Descent
├── GDRegressor.py
├── derivation.md
├── example.py
└── README.md
```

---

## Example Usage

```python
from gradient_descent import GradientDescent
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X, y = make_regression(
    n_samples=500,
    n_features=2,
    noise=10,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = GradientDescent(
    learning_rate=0.01,
    epochs=1000
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(model.score(X_test, y_test))
```

---

## Time Complexity

| Algorithm | Training Complexity |
|-----------|--------------------|
| Batch GD | O(epochs × n_samples × n_features) |
| SGD | O(epochs × n_samples × n_features) |
| Mini-Batch GD | O(epochs × n_samples × n_features) |

Although the asymptotic complexity is the same, **SGD** and **Mini-Batch GD** perform much more frequent parameter updates and are generally preferred for large datasets.

### Prediction

```
O(n_samples × n_features)
```

### Space Complexity

```
O(n_features)
```

---

## Requirements

- Python 3.x
- NumPy

Install NumPy

```bash
pip install numpy
```

---



## License

This project is intended for educational purposes to demonstrate how Linear Regression optimization algorithms work internally.
