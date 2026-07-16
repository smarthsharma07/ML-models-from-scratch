# Machine Learning Models From Scratch

A collection of Machine Learning algorithms implemented completely from scratch using **NumPy**.

The goal of this repository is to understand the mathematics, optimization techniques, and implementation details behind machine learning algorithms instead of relying solely on high-level libraries such as **scikit-learn**.

Every implementation is built from first principles and includes mathematical derivations, clean code, practical examples, and comparisons with established libraries whenever applicable.

---

# Objectives

- Understand the mathematical foundations of Machine Learning.
- Implement machine learning algorithms entirely from scratch using NumPy.
- Learn optimization algorithms and the mathematics behind them.
- Build reusable, modular, and well-documented implementations.
- Follow a clean API inspired by scikit-learn.
- Validate implementations against scikit-learn whenever possible.
- Create an educational resource for learning, experimentation, and reference.

---

# Repository Structure

```text
ml-models-from-scratch/
│
├── supervised_learning/
│   │
│   ├── simple_linear_regression/
│   │   ├── linear_regression.py
│   │   ├── derivation.md
│   │   ├── example.py
│   │   └── README.md
│   │
│   ├── multiple_linear_regression/
│   │   ├── multiple_linear_regression.py
│   │   ├── derivation.md
│   │   ├── example.py
│   │   └── README.md
│   │
│   ├── polynomial_linear_regression/
│   │   ├── polynomial_linear_regression.py
│   │   ├── derivation.md
│   │   ├── example.py
│   │   └── README.md
│   │
│   ├── ridge_regression/
│   │   ├── ridge_regression.py
│   │   ├── derivation.md
│   │   ├── example.py
│   │   └── README.md
│   │
│   └── gradient_descent_scratch/
│       ├── gradient_descent.py
│       ├── stochastic_gradient_descent.py
│       ├── mini_batch_gradient_descent.py
│       ├── derivation.md
│       ├── example.py
│       └── README.md
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

# Implemented Algorithms

## Supervised Learning

### Regression

| Algorithm | Status |
|-----------|:------:|
| Simple Linear Regression | ✅ |
| Multiple Linear Regression | ✅ |
| Polynomial Linear Regression | ✅ |
| Ridge Regression | ✅ |

### Optimization Algorithms

| Algorithm | Status |
|-----------|:------:|
| Batch Gradient Descent | ✅ |
| Stochastic Gradient Descent (SGD) | ✅ |
| Mini-Batch Gradient Descent | ✅ |

---

# Optimization Algorithms

The repository currently includes implementations of:

- ✅ Batch Gradient Descent
- ✅ Stochastic Gradient Descent (SGD)
- ✅ Mini-Batch Gradient Descent

These optimizers are implemented independently to demonstrate how machine learning models learn parameters through iterative optimization.

---

# Roadmap

## Supervised Learning

### Regression

- ✅ Simple Linear Regression
- ✅ Multiple Linear Regression
- ✅ Polynomial Linear Regression
- ✅ Ridge Regression
- Lasso Regression
- Elastic Net Regression
- Logistic Regression

### Classification

- K-Nearest Neighbors (KNN)
- Naive Bayes
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest

---

## Unsupervised Learning

- K-Means Clustering
- DBSCAN
- Hierarchical Clustering
- Gaussian Mixture Models (GMM)

---

## Dimensionality Reduction

- Principal Component Analysis (PCA)
- Linear Discriminant Analysis (LDA)
- t-SNE

---

## Deep Learning

- Perceptron
- Feed Forward Neural Network
- Backpropagation
- Activation Functions
- Loss Functions
- Optimizers
- Convolutional Neural Networks (CNNs)

---

# Features

- Pure NumPy implementations
- No machine learning libraries used in model implementations
- Mathematical derivations for every algorithm
- Step-by-step implementations from first principles
- Modular and reusable codebase
- Clean API inspired by scikit-learn
- Practical examples for every algorithm
- Validation against scikit-learn implementations
- Comprehensive documentation
- Educational focus with readable and maintainable code
- Repository designed for continuous expansion

---

# Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/ml-models-from-scratch.git
```

Move into the project

```bash
cd ml-models-from-scratch
```

Install the dependencies

```bash
pip install -r requirements.txt
```

---

# Example

```python
from supervised_learning.ridge_regression.ridge_regression import RidgeRegression

model = RidgeRegression(
    learning_rate=0.01,
    epochs=1000,
    alpha=1.0
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Weights:", model.weights)
print("Bias:", model.bias)
print("R² Score:", model.score(X_test, y_test))
```

---

# Validation

Wherever applicable, implementations are compared against equivalent models from **scikit-learn** to verify mathematical correctness and numerical accuracy.

Minor differences may occur because of floating-point precision, optimization strategies, initialization methods, or implementation details.

---

# Future Improvements

- Feature Scaling (StandardScaler, MinMaxScaler)
- Cross Validation
- Evaluation Metrics
- Learning Rate Scheduling
- Early Stopping
- Hyperparameter Tuning
- Model Serialization
- Unit Tests
- Benchmarking
- Visualization Utilities
- Performance Profiling
- Comprehensive Documentation

---

# Contributing

Contributions, suggestions, bug reports, and feature requests are welcome.

If you find an issue or have an idea for improvement, feel free to open an issue or submit a pull request.

---

# License

This project is licensed under the **MIT License**.

---

# Acknowledgements

This repository is an educational project created to deepen my understanding of machine learning by implementing algorithms from first principles. The focus is on understanding the underlying mathematics, optimization techniques, and implementation details rather than relying on high-level machine learning libraries.
