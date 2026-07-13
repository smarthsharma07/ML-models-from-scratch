from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

from gradient_descent import GradientDescent

X, y = make_regression(
    n_samples=500,
    n_features=2,
    noise=10,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
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