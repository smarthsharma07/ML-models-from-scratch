# Machine Learning Models From Scratch

A collection of Machine Learning algorithms implemented from scratch using **NumPy**.

The purpose of this repository is to understand the mathematics, algorithms, and implementation details behind machine learning models instead of relying solely on high-level libraries such as **scikit-learn**.

Every implementation focuses on:
- Understanding the underlying mathematics
- Writing clean, reusable code
- Validating results against scikit-learn (where applicable)

---

## Objectives

- Learn the mathematical foundations of Machine Learning.
- Implement popular algorithms from scratch using NumPy.
- Maintain a clean and consistent API inspired by scikit-learn.
- Compare implementations with scikit-learn for correctness.
- Build an educational machine learning library for learning and reference.

---

## Repository Structure

```text
ml-models-from-scratch/
│
├── supervised_learning/
│   ├── simple_linear_regression/
│   │   ├── linear_regression.py
│   │   ├── example.py
│   │   └── README.md
│   │
│   └── multiple_linear_regression/
│       ├── multiple_linear_regression.py
│       ├── example.py
│       └── README.md
│
├── requirements.txt
└── README.md
```

---

## Implemented Algorithms

### Supervised Learning

- ✅ Simple Linear Regression
- ✅ Multiple Linear Regression (Normal Equation)

---

## Roadmap

### Supervised Learning

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Naive Bayes

### Unsupervised Learning

- K-Means Clustering

### Dimensionality Reduction

- Principal Component Analysis (PCA)

### Deep Learning

- Feed Forward Neural Network

---

## Features

- Pure NumPy implementations
- Clean and beginner-friendly code
- Mathematical explanations
- Usage examples
- Consistent API across models
- Validation against scikit-learn

---

## Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/ml-models-from-scratch.git
```

Install the required dependencies

```bash
pip install -r requirements.txt
```

---

## Example

```python
from supervised_learning.simple_linear_regression.linear_regression import LinearRegressionScratch

model = LinearRegressionScratch()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(model.weight)
print(model.bias)
```

---

## Validation

Each implementation is compared against the equivalent implementation in **scikit-learn** whenever possible to verify correctness. Minor numerical differences may occur because of floating-point precision.

---

## Future Improvements

- Gradient Descent implementations
- Mini-Batch Gradient Descent
- Ridge Regression
- Lasso Regression
- Polynomial Regression
- Feature Scaling utilities
- Unit tests
- Performance benchmarks
- Interactive notebooks
- Comprehensive documentation for every algorithm

---

## Contributing

Contributions, suggestions, and bug reports are welcome. Feel free to open an issue or submit a pull request.

---

## License

This project is licensed under the **MIT License**.
