# Derivation of Simple Linear Regression

## Objective

Given a dataset

$$
(x_1,y_1),(x_2,y_2),\ldots,(x_n,y_n)
$$

we wish to find the best fitting line

$$
\hat{y}=wx+b
$$

that minimizes the Mean Squared Error.

---

## Cost Function

The loss is

$$
J(w,b)=\sum_{i=1}^{n}(y_i-(wx_i+b))^2
$$

---

## Partial derivative with respect to \(b\)

Setting

$$
\frac{\partial J}{\partial b}=0
$$

gives

$$
b=\bar{y}-w\bar{x}
$$

---

## Partial derivative with respect to \(w\)

Substituting the above expression and simplifying,

$$
w=
\frac
{\sum (x_i-\bar{x})(y_i-\bar{y})}
{\sum (x_i-\bar{x})^2}
$$

---

## Final Model

$$
\boxed{
\hat{y}=wx+b
}
$$
