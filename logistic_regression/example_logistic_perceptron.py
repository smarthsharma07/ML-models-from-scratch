import numpy as np
from sklearn.datasets import make_classification
from logistic_regression_perceptron_trick import LogisticRegression_Perceptron as LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create dataset
X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    random_state=42
)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train
model = LogisticRegression(
    learning_rate=0.1,
    epochs=1000
)

model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print("Weights:", model.weights)
print("Bias:", model.bias)