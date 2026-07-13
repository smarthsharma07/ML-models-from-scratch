# Linear Regression using Gradient Descent (From Scratch)

A NumPy implementation of **Linear Regression** trained using **Batch Gradient Descent**, built completely from scratch without using machine learning libraries such as scikit-learn.

## Features

* Batch Gradient Descent
* Learnable weights and bias
* Mean Squared Error (MSE) loss
* Configurable learning rate
* Configurable number of epochs
* Early stopping using gradient tolerance
* Prediction on unseen data
* R² score evaluation
* Loss history for visualizing convergence

## Mathematical Model

For a dataset with one or more features, the prediction is given by

[
\hat{y} = Xw + b
]

where:

* (X) = feature matrix
* (w) = weight vector
* (b) = bias
* (\hat{y}) = predicted output

The objective is to minimize the Mean Squared Error (MSE) between the predicted and actual values.

## Gradient Descent

Instead of solving the normal equation directly, this implementation minimizes the loss iteratively using Gradient Descent.

At every epoch:

1. Compute predictions.
2. Calculate prediction errors.
3. Compute gradients of the loss with respect to the weights and bias.
4. Update the parameters in the direction of decreasing loss.

The process repeats until the specified number of epochs is reached or the gradients become sufficiently small.

## Usage

```python
from gradient_descent import GradientDescent
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

X, y = make_regression(
    n_samples=500,
    n_features=2,
    noise=10,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = GradientDescent(
    learning_rate=0.01,
    epochs=1000
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Weights:", model.weights)
print("Bias:", model.bias)
print("R² Score:", model.score(X_test, y_test))
```

## Time Complexity

### Training

Each epoch performs matrix multiplication over all samples and features.

**Time Complexity:** `O(epochs × n_samples × n_features)`

### Prediction

**Time Complexity:** `O(n_samples × n_features)`

### Space Complexity

**Space Complexity:** `O(n_features)`

## Requirements

* Python 3.x
* NumPy

Install NumPy using:

```bash
pip install numpy
```

## Future Improvements

* Stochastic Gradient Descent (SGD)
* Mini-Batch Gradient Descent
* L1/L2 Regularization
* Momentum
* Adam Optimizer
* Learning Rate Scheduling
* Feature Scaling utilities

## License

This project is intended for educational purposes to demonstrate how Linear Regression and Gradient Descent work internally.
