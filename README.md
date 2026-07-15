# Machine Learning Models From Scratch

A collection of Machine Learning algorithms implemented completely from scratch using **NumPy**.

The goal of this repository is to understand the mathematics, optimization techniques, and implementation details behind machine learning algorithms instead of relying solely on high-level libraries such as **scikit-learn**.

Every implementation is built from first principles and includes mathematical derivations, clean code, practical examples, and comparisons with established libraries whenever applicable.

---

# Objectives

- Understand the mathematical foundations of Machine Learning.
- Implement popular machine learning algorithms from scratch using NumPy.
- Learn optimization algorithms used in machine learning.
- Build reusable and well-structured implementations.
- Maintain a clean API inspired by scikit-learn.
- Validate implementations against scikit-learn whenever possible.
- Create an educational repository for learning and reference.

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

### Linear Models

- ✅ Simple Linear Regression (Normal Equation)
- ✅ Multiple Linear Regression (Normal Equation)
- ✅ Batch Gradient Descent for Linear Regression
- ✅ Stochastic Gradient Descent (SGD)
- ✅ Mini-Batch Gradient Descent

---

# Optimization Algorithms

Implemented optimizers

- ✅ Batch Gradient Descent
- ✅ Stochastic Gradient Descent
- ✅ Mini-Batch Gradient Descent

# Roadmap

## Supervised Learning

- Logistic Regression
- Ridge Regression
- Lasso Regression
- Elastic Net Regression
- Polynomial Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Naive Bayes

## Unsupervised Learning

- K-Means Clustering
- DBSCAN
- Hierarchical Clustering

## Dimensionality Reduction

- Principal Component Analysis (PCA)
- Linear Discriminant Analysis (LDA)

## Deep Learning

- Perceptron
- Feed Forward Neural Network
- Backpropagation
- Activation Functions
- Loss Functions
- Optimizers

---

# Features

- Pure NumPy implementations
- No machine learning libraries used in model implementations
- Mathematical derivations for every algorithm
- Clean, modular, and well-documented code
- Example scripts for every model
- Consistent API inspired by scikit-learn
- Validation against scikit-learn implementations
- Educational focus with readable implementations
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

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Example

```python
from supervised_learning.gradient_descent_scratch.gradient_descent import GradientDescent

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

---

# Validation

Wherever possible, implementations are compared against equivalent models from **scikit-learn** to verify correctness.

Minor numerical differences may occur due to floating-point precision, parameter initialization, or optimization strategies.

---

# Future Improvements

- Feature Scaling (StandardScaler, MinMaxScaler)
- Data Preprocessing Utilities
- Cross Validation
- Evaluation Metrics
- Regularization Techniques
- Learning Rate Scheduling
- Early Stopping
- Model Serialization
- Unit Tests
- Benchmarking
- Visualization Utilities
- Comprehensive Documentation

---

# Contributing

Contributions, suggestions, bug reports, and feature requests are welcome.

If you find an issue or have an improvement in mind, feel free to open an issue or submit a pull request.

---

# License

This project is licensed under the **MIT License**.

---

## Acknowledgements

This repository is intended for educational purposes and serves as a personal learning journey into implementing machine learning algorithms from first principles.
