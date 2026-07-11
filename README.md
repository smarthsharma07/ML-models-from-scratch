# Machine Learning Models From Scratch

A collection of Machine Learning algorithms implemented from scratch using **NumPy**.

The goal of this repository is to understand the mathematics and implementation details behind popular machine learning algorithms instead of relying on high-level libraries like scikit-learn.

## Objectives

- Understand the mathematical foundations of ML algorithms.
- Implement each algorithm from scratch.
- Maintain a clean and consistent API across models.
- Compare implementations with scikit-learn for correctness.
- Build a personal educational machine learning library.

---

## Repository Structure

```
ml-models-from-scratch/
│
├── Simple Linear Regression/
│   ├── linear_regression.py
│   ├── example.py
│   └── README.md
│
├── Logistic Regression/
├── K-Nearest Neighbors/
├── Decision Tree/
├── Random Forest/
├── Support Vector Machine/
├── Naive Bayes/
├── K-Means/
├── PCA/
├── Neural Network/
│
├── requirements.txt
└── README.md
```

---

## Implemented Algorithms

| Algorithm | Status |
|-----------|--------|
| Simple Linear Regression | ✅ |
| Multiple Linear Regression | ⏳ |
| Logistic Regression | ⏳ |
| K-Nearest Neighbors | ⏳ |
| Decision Tree | ⏳ |
| Random Forest | ⏳ |
| Naive Bayes | ⏳ |
| Support Vector Machine | ⏳ |
| K-Means Clustering | ⏳ |
| Principal Component Analysis | ⏳ |
| Neural Network | ⏳ |

---

## Features

- Pure NumPy implementations
- Well-commented code
- Mathematical explanations
- Usage examples
- Simple and readable API

---

## Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/ml-models-from-scratch.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Example

```python
from linear_regression import LinearRegressionScratch

model = LinearRegressionScratch()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(model.weight)
print(model.bias)
```

---

## Future Improvements

- Unit tests
- Performance benchmarks
- Gradient Descent implementations
- Visualization notebooks
- Feature scaling utilities
- Regularization (Lasso & Ridge)

---

## License

This project is licensed under the MIT License.
