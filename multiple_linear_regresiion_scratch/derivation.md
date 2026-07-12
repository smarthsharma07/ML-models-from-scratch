# Derivation of Multiple Linear Regression

## Model

The hypothesis is

$$
\hat{y}=Xw
$$

where

- \(X\) is the feature matrix.
- \(w\) is the weight vector.

---

## Cost Function

The objective is to minimize

$$
J(w)=
(y-Xw)^T(y-Xw)
$$

---

## Expand the expression

$$
J(w)
=
y^Ty
-
2w^TX^Ty
+
w^TX^TXw
$$

---

## Differentiate

Taking the derivative with respect to \(w\),

$$
\frac{\partial J}{\partial w}
=
-2X^Ty
+
2X^TXw
$$

---

## Set derivative equal to zero

$$
-2X^Ty
+
2X^TXw
=
0
$$

Rearranging,

$$
X^TXw=X^Ty
$$

---

## Solve for the weights

Multiplying both sides by

$$
(X^TX)^{-1}
$$

gives

$$
\boxed{
w=(X^TX)^{-1}X^Ty
}
$$

This expression is known as the **Normal Equation**.
