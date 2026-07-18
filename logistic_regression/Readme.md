# Logistic Regression from Scratch

A complete implementation of **Binary Logistic Regression** from scratch using **NumPy**.

This project demonstrates how logistic regression learns a linear decision boundary by estimating the probability that a sample belongs to the positive class using the **Sigmoid Activation Function**. The model is trained using **Batch Gradient Descent** without relying on machine learning libraries such as **scikit-learn** for the implementation.

---

## Features

* Binary classification
* Implemented entirely using NumPy
* Batch Gradient Descent optimization
* Sigmoid activation function
* Probability prediction (`predict_proba`)
* Binary class prediction (`predict`)
* Simple and well-documented code

---

## Mathematical Background

Given an input vector

[
x = [x_1, x_2, ..., x_n]
]

the model computes the linear combination

[
z = w^Tx + b
]

where

* (w) = weight vector
* (b) = bias

The output probability is obtained using the **Sigmoid Function**

[
\sigma(z)=\frac{1}{1+e^{-z}}
]

which maps any real number into the range

[
0 \leq \sigma(z) \leq 1
]

The prediction rule is

[
\hat y=
\begin{cases}
1,&\sigma(z)\ge0.5\
0,&\sigma(z)<0.5
\end{cases}
]

---

## Training

The model is trained using **Batch Gradient Descent**.

### Forward Pass

Compute

[
z = Xw+b
]

Apply sigmoid

[
\hat y=\sigma(z)
]

---

### Gradients

Weight gradient

[
\frac{\partial J}{\partial w}
=============================

\frac1mX^T(\hat y-y)
]

Bias gradient

[
\frac{\partial J}{\partial b}
=============================

\frac1m\sum(\hat y-y)
]

---

### Parameter Update

Weights

[
w=w-\alpha\frac{\partial J}{\partial w}
]

Bias

[
b=b-\alpha\frac{\partial J}{\partial b}
]

where

* (m) = number of training samples
* (\alpha) = learning rate

---

## Project Structure

```
LogisticRegression/
│
├── logistic_regression.py
├── example.py
└──  README.md
```

---

## Example

```python
from logistic_regression import LogisticRegression

model = LogisticRegression(
    learning_rate=0.1,
    epochs=1000
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = np.mean(predictions == y_test)

print("Accuracy:", accuracy)
```

---

## Methods

| Method             | Description                     |
| ------------------ | ------------------------------- |
| `fit(X, y)`        | Train the model                 |
| `predict_proba(X)` | Return predicted probabilities  |
| `predict(X)`       | Return binary class predictions |
| `sigmoid(z)`       | Compute sigmoid activation      |

---

## Time Complexity

### Training

For each epoch

[
O(mn)
]

where

* (m) = number of samples
* (n) = number of features

Overall

[
O(\text{epochs} \times m \times n)
]

### Prediction

[
O(mn)
]

---

## Requirements

* Python 3.x
* NumPy

Install dependencies

```bash
pip install numpy
```

---

## Future Improvements

* Mini-Batch Gradient Descent
* Stochastic Gradient Descent
* L2 Regularization (Ridge)
* L1 Regularization (Lasso)
* Early Stopping
* Decision Boundary Visualization
* Multiclass Logistic Regression (Softmax Regression)
* Training Loss Tracking
* Numerical Stability Improvements

---

## References

* Christopher M. Bishop — *Pattern Recognition and Machine Learning*
* Tom M. Mitchell — *Machine Learning*
* Andrew Ng — *Machine Learning Specialization*
* *The Elements of Statistical Learning* by Hastie, Tibshirani, and Friedman

---

## License

This project is intended for educational purposes to understand the mathematical foundations and implementation details of Logistic Regression from first principles.
