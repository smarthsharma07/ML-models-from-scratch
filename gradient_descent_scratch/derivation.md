# Derivation of Gradient Descent for Linear Regression

## 1. Hypothesis

For a dataset with (n) samples and (m) features, the prediction is

[
\hat{y}=Xw+b
]

where

* (X) is the feature matrix
* (w) is the weight vector
* (b) is the bias
* (\hat{y}) is the predicted output

---

## 2. Loss Function

The objective is to minimize the Mean Squared Error (MSE).

[
J(w,b)=\frac{1}{2n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
]

The factor of (\frac{1}{2}) simplifies differentiation because the derivative of the square introduces a factor of 2, which cancels it.

Substituting the prediction equation,

[
J(w,b)=\frac{1}{2n}\sum_{i=1}^{n}(y_i-(Xw+b))^2
]

---

## 3. Gradient with Respect to the Weights

Differentiate the loss with respect to the weight vector.

[
\frac{\partial J}{\partial w}
=============================

\frac{1}{n}X^T(\hat{y}-y)
]

This gradient indicates how each weight influences the loss.

---

## 4. Gradient with Respect to the Bias

Differentiate the loss with respect to the bias.

[
\frac{\partial J}{\partial b}
=============================

\frac{1}{n}\sum(\hat{y}-y)
]

Since the bias affects every prediction equally, its gradient is simply the average prediction error.

---

## 5. Gradient Descent Update Rule

Gradient Descent updates the parameters by moving in the direction opposite to the gradient.

### Update the weights

[
w
=

## w

\alpha
\frac{\partial J}{\partial w}
]

Substituting the gradient,

[
w
=

## w

\alpha
\left(
\frac{1}{n}
X^T
(\hat{y}-y)
\right)
]

---

### Update the bias

[
b
=

## b

\alpha
\frac{\partial J}{\partial b}
]

Substituting the gradient,

[
b
=

## b

\alpha
\left(
\frac{1}{n}
\sum(\hat{y}-y)
\right)
]

where

* (\alpha) is the learning rate.

---

## 6. Training Algorithm

Repeat the following steps for each epoch:

1. Compute predictions.

[
\hat{y}=Xw+b
]

2. Compute prediction errors.

[
\text{error}=\hat{y}-y
]

3. Compute the gradients.

[
dw=\frac{1}{n}X^T(\hat{y}-y)
]

[
db=\frac{1}{n}\sum(\hat{y}-y)
]

4. Update the parameters.

[
w=w-\alpha dw
]

[
b=b-\alpha db
]

5. Repeat until the desired number of epochs is reached or the gradients become sufficiently small.

---

## Corresponding NumPy Implementation

```python
predictions = np.dot(X_train, self.weights) + self.bias

errors = predictions - y_train

dw = (1 / n_samples) * np.dot(X_train.T, errors)
db = (1 / n_samples) * np.sum(errors)

self.weights -= self.learning_rate * dw
self.bias -= self.learning_rate * db
```

These equations form the mathematical foundation of Batch Gradient Descent for Linear Regression.
