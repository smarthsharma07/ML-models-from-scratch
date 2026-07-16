# Machine Learning Models From Scratch

A collection of Machine Learning algorithms implemented completely from scratch using **NumPy**.

The purpose of this repository is to understand the mathematics, optimization algorithms, and implementation details behind machine learning models instead of relying solely on high-level libraries such as **scikit-learn**.

Each implementation is built from first principles and includes mathematical derivations, clean code, practical examples, and validation against established libraries whenever applicable.

---

# Objectives

* Understand the mathematical foundations of Machine Learning.
* Implement machine learning algorithms from scratch using NumPy.
* Learn optimization algorithms and their underlying mathematics.
* Build reusable, modular, and well-documented implementations.
* Maintain a clean API inspired by scikit-learn.
* Validate implementations against scikit-learn whenever possible.
* Create an educational repository for learning, experimentation, and reference.

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

| Algorithm                         | Status |
| --------------------------------- | :----: |
| Simple Linear Regression          |    ✅   |
| Multiple Linear Regression        |    ✅   |
| Polynomial Linear Regression      |    ✅   |
| Batch Gradient Descent            |    ✅   |
| Stochastic Gradient Descent (SGD) |    ✅   |
| Mini-Batch Gradient Descent       |    ✅   |

---

# Optimization Algorithms

The repository currently includes implementations of the following optimization algorithms:

* ✅ Batch Gradient Descent
* ✅ Stochastic Gradient Descent (SGD)
* ✅ Mini-Batch Gradient Descent

---

# Roadmap

## Supervised Learning

### Regression

* ✅ Simple Linear Regression
* ✅ Multiple Linear Regression
* ✅ Polynomial Linear Regression
* Logistic Regression
* Ridge Regression
* Lasso Regression
* Elastic Net Regression

### Classification

* K-Nearest Neighbors (KNN)
* Logistic Regression
* Naive Bayes
* Support Vector Machine (SVM)
* Decision Tree
* Random Forest

---

## Unsupervised Learning

* K-Means Clustering
* DBSCAN
* Hierarchical Clustering
* Gaussian Mixture Models (GMM)

---

## Dimensionality Reduction

* Principal Component Analysis (PCA)
* Linear Discriminant Analysis (LDA)
* t-SNE

---

## Deep Learning

* Perceptron
* Feed Forward Neural Network
* Backpropagation
* Activation Functions
* Loss Functions
* Optimizers
* Convolutional Neural Networks (CNNs)

---

# Features

* Pure NumPy implementations
* No machine learning libraries used in model implementations
* Mathematical derivations for every algorithm
* Step-by-step implementations from first principles
* Modular and reusable codebase
* Clean API inspired by scikit-learn
* Practical examples for every algorithm
* Validation against scikit-learn implementations
* Extensive documentation
* Educational focus with readable code
* Repository designed for continuous expansion

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

Install the required dependencies

```bash
pip install -r requirements.txt
```

---

# Example

```python
from supervised_learning.polynomial_linear_regression.polynomial_linear_regression import PolynomialLinearRegression

model = PolynomialLinearRegression(degree=2)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Weights:", model.weights)
print("R² Score:", model.score(X_test, y_test))
```

---

# Validation

Wherever applicable, implementations are compared with equivalent models from **scikit-learn** to verify mathematical correctness and numerical accuracy.

Minor differences may occur due to floating-point precision, implementation details, initialization strategies, or optimization methods.

---

# Future Improvements

* Feature Scaling (StandardScaler, MinMaxScaler)
* Cross Validation
* Evaluation Metrics
* Regularization Techniques
* Learning Rate Scheduling
* Early Stopping
* Hyperparameter Tuning
* Model Serialization
* Unit Tests
* Benchmarking
* Visualization Utilities
* Performance Profiling
* Comprehensive Documentation

---

# Contributing

Contributions, suggestions, bug reports, and feature requests are welcome.

If you discover an issue or have an idea for improvement, feel free to open an issue or submit a pull request.

---

# License

This project is licensed under the **MIT License**.

---

# Acknowledgements

This repository is an educational project created to deepen my understanding of machine learning by implementing algorithms from first principles. The emphasis is on understanding the underlying mathematics, optimization techniques, and implementation details rather than relying on high-level machine learning libraries.
