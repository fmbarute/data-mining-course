import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score

# (a) Generate dataset with quadratic decision boundary
rng = np.random.default_rng(5)
x1 = rng.uniform(size=500) - 0.5
x2 = rng.uniform(size=500) - 0.5
y = (x1**2 - x2**2 > 0).astype(int)
X = np.column_stack((x1, x2))

# (b) Plot original data
plt.figure(figsize=(12, 10))
plt.subplot(3, 2, 1)
plt.scatter(x1, x2, c=y, cmap='coolwarm', alpha=0.7)
plt.title("(b) Original Data with True Classes")
plt.xlabel("X1")
plt.ylabel("X2")

# (c) Fit basic logistic regression
logreg_linear = LogisticRegression()
logreg_linear.fit(X, y)

# (d) Plot linear logistic regression predictions
y_pred_linear = logreg_linear.predict(X)
plt.subplot(3, 2, 2)
plt.scatter(x1, x2, c=y_pred_linear, cmap='coolwarm', alpha=0.7)
plt.title("(d) Linear Logistic Regression Predictions")
plt.xlabel("X1")
plt.ylabel("X2")

# (e) Fit logistic regression with polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
logreg_poly = LogisticRegression(max_iter=1000)
logreg_poly.fit(X_poly, y)

# (f) Plot polynomial logistic regression predictions
y_pred_poly = logreg_poly.predict(X_poly)
plt.subplot(3, 2, 3)
plt.scatter(x1, x2, c=y_pred_poly, cmap='coolwarm', alpha=0.7)
plt.title("(f) Polynomial Logistic Regression Predictions")
plt.xlabel("X1")
plt.ylabel("X2")

# (g) Fit linear SVM
svm_linear = SVC(kernel='linear')
svm_linear.fit(X, y)

# Plot linear SVM predictions
y_pred_svm_linear = svm_linear.predict(X)
plt.subplot(3, 2, 4)
plt.scatter(x1, x2, c=y_pred_svm_linear, cmap='coolwarm', alpha=0.7)
plt.title("(g) Linear SVM Predictions")
plt.xlabel("X1")
plt.ylabel("X2")

# (h) Fit SVM with RBF kernel
svm_rbf = SVC(kernel='rbf', gamma=1)
svm_rbf.fit(X, y)

# Plot RBF SVM predictions
y_pred_svm_rbf = svm_rbf.predict(X)
plt.subplot(3, 2, 5)
plt.scatter(x1, x2, c=y_pred_svm_rbf, cmap='coolwarm', alpha=0.7)
plt.title("(h) RBF Kernel SVM Predictions")
plt.xlabel("X1")
plt.ylabel("X2")

plt.tight_layout()
plt.show()

# (i) Results comparison
print("\nModel Accuracies:")
print(f"Linear Logistic Regression: {accuracy_score(y, y_pred_linear):.4f}")
print(f"Polynomial Logistic Regression: {accuracy_score(y, y_pred_poly):.4f}")
print(f"Linear SVM: {accuracy_score(y, y_pred_svm_linear):.4f}")
print(f"RBF Kernel SVM: {accuracy_score(y, y_pred_svm_rbf):.4f}")