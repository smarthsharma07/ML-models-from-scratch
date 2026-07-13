# Machine Learning Models From Scratch

A collection of Machine Learning algorithms implemented from scratch using **NumPy**.

The purpose of this repository is to understand the mathematics, algorithms, and implementation details behind machine learning models instead of relying solely on high-level libraries such as **scikit-learn**.

Each implementation is built from first principles, accompanied by mathematical derivations, clean code, and practical examples.

---

## Objectives

* Learn the mathematical foundations of Machine Learning.
* Implement popular machine learning algorithms from scratch using NumPy.
* Build reusable, well-structured implementations.
* Maintain a clean API inspired by scikit-learn.
* Validate implementations against scikit-learn whenever applicable.
* Create an educational repository for learning and reference.

---

## Repository Structure

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
│   └── gradient_descent/
│       ├── gradient_descent.py
│       ├── derivation.md
│       ├── example.py
│       └── README.md
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Implemented Algorithms

### Supervised Learning

* ✅ Simple Linear Regression (Normal Equation)
* ✅ Multiple Linear Regression (Normal Equation)
* ✅ Linear Regression using Batch Gradient Descent

---

## Roadmap

### Supervised Learning

* Logistic Regression
* Ridge Regression
* Lasso Regression
* Polynomial Regression
* K-Nearest Neighbors (KNN)
* Decision Tree
* Random Forest
* Support Vector Machine (SVM)
* Naive Bayes

### Optimization

* Stochastic Gradient Descent (SGD)
* Mini-Batch Gradient Descent
* Momentum Gradient Descent
* RMSProp
* Adam Optimizer

### Unsupervised Learning

* K-Means Clustering

### Dimensionality Reduction

* Principal Component Analysis (PCA)

### Deep Learning

* Feed Forward Neural Network
* Backpropagation
* Activation Functions
* Loss Functions

---

## Features

* Pure NumPy implementations
* No machine learning libraries used in model implementations
* Mathematical derivations for every algorithm
* Beginner-friendly and well-documented code
* Example scripts for every model
* Consistent API inspired by scikit-learn
* Validation against scikit-learn implementations
* Modular project structure for easy extension

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/ml-models-from-scratch.git
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Example

```python
from supervised_learning.gradient_descent.gradient_descent import GradientDescent

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

## Validation

Whenever possible, each implementation is validated against its equivalent implementation in **scikit-learn**. Small numerical differences may occur because of floating-point precision or optimization methods.

---

## Future Improvements

* Feature Scaling (StandardScaler, MinMaxScaler)
* Cross Validation
* Evaluation Metrics
* Regularization Techniques
* Model Serialization
* Unit Tests
* Performance Benchmarks
* Visualization Utilities
* Interactive Jupyter Notebooks
* Comprehensive Documentation

---

## Contributing

Contributions, suggestions, and bug reports are welcome. Feel free to open an issue or submit a pull request.

---

## License

This project is licensed under the **MIT License**.
