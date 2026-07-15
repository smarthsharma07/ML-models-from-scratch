# Derivation of Gradient Descent for Linear Regression

This document derives the Gradient Descent optimization algorithm used to train a Linear Regression model.

---

# 1. Linear Regression Hypothesis

Given a dataset with **n** samples and **m** features,

- $X \in \mathbb{R}^{n \times m}$ : Feature matrix
- $w \in \mathbb{R}^{m}$ : Weight vector
- $b \in \mathbb{R}$ : Bias
- $\hat{y}$ : Predicted output

The prediction equation is

$$
\hat{y}=Xw+b
$$

---

# 2. Loss Function

To measure how well the model fits the data, we use the **Mean Squared Error (MSE)** loss.

$$
J(w,b)=\frac{1}{2n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

The factor $\frac{1}{2}$ is included because it cancels the factor of 2 that appears during differentiation.

Substituting the hypothesis,

$$
J(w,b)=\frac{1}{2n}\sum_{i=1}^{n}\left(y_i-(Xw+b)\right)^2
$$

---

# 3. Gradient with Respect to the Weights

To minimize the loss, compute the partial derivative with respect to the weight vector.

Using the chain rule,

$$
\frac{\partial J}{\partial w}
=
\frac{1}{n}X^T(\hat{y}-y)
$$

This gradient tells us how changing each weight affects the loss.

---

# 4. Gradient with Respect to the Bias

Similarly,

$$
\frac{\partial J}{\partial b}
=
\frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i-y_i)
$$

Since the bias contributes equally to every prediction, its gradient is simply the average prediction error.

---

# 5. Gradient Descent Update Rule

Gradient Descent updates the parameters in the direction opposite to the gradient.

### Update the weights

$$
w
=
w
-
\alpha
\frac{\partial J}{\partial w}
$$

Substituting the gradient,

$$
w
=
w
-
\alpha
\left(
\frac{1}{n}
X^T(\hat{y}-y)
\right)
$$

---

### Update the bias

$$
b
=
b
-
\alpha
\frac{\partial J}{\partial b}
$$

Substituting,

$$
b
=
b
-
\alpha
\left(
\frac{1}{n}
\sum_{i=1}^{n}
(\hat{y}_i-y_i)
\right)
$$

where

- $\alpha$ = Learning Rate

---

# 6. Complete Gradient Descent Algorithm

For each training epoch,

### Step 1: Compute Predictions

$$
\hat{y}=Xw+b
$$

---

### Step 2: Compute Errors

$$
\text{error}
=
\hat{y}-y
$$

---

### Step 3: Compute Gradients

Weight gradient

$$
dw
=
\frac{1}{n}
X^T(\hat{y}-y)
$$

Bias gradient

$$
db
=
\frac{1}{n}
\sum_{i=1}^{n}
(\hat{y}-y)
$$

---

### Step 4: Update Parameters

Weights

$$
w
=
w-\alpha dw
$$

Bias

$$
b
=
b-\alpha db
$$

---

### Step 5

Repeat until

- the desired number of epochs is reached, or
- the gradients become sufficiently small.

---

# Corresponding NumPy Implementation

```python
predictions = np.dot(X_train, self.weights) + self.bias

errors = predictions - y_train

dw = (1 / n_samples) * np.dot(X_train.T, errors)
db = (1 / n_samples) * np.sum(errors)

self.weights -= self.learning_rate * dw
self.bias -= self.learning_rate * db
```

---

# Time Complexity

For each epoch,

- Prediction: **O(n × m)**
- Gradient Computation: **O(n × m)**
- Parameter Update: **O(m)**

Overall training complexity:

$$
O(\text{epochs} \times n \times m)
$$

where

- $n$ = Number of samples
- $m$ = Number of features

---

This derivation forms the mathematical foundation of **Batch Gradient Descent for Linear Regression**. The same objective function is also optimized by **Stochastic Gradient Descent (SGD)** and **Mini-Batch Gradient Descent**, which differ only in the number of training samples used to compute the gradients at each update.
