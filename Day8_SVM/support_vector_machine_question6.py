import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

# Set random seed for reproducibility
np.random.seed(42)


# (a) Generate barely linearly separable data
def generate_barely_separable_data(n=100):
    # Generate class 0 (centered at (-0.5, -0.5))
    X0 = np.random.multivariate_normal(mean=[-0.5, -0.5],
                                       cov=[[0.05, 0], [0, 0.05]],
                                       size=n // 2)
    # Generate class 1 (centered at (0.5, 0.5))
    X1 = np.random.multivariate_normal(mean=[0.5, 0.5],
                                       cov=[[0.05, 0], [0, 0.05]],
                                       size=n // 2)

    X = np.vstack([X0, X1])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    return X, y


X_train, y_train = generate_barely_separable_data(100)

# Plot the training data
plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', alpha=0.7)
plt.title("(a) Barely Linearly Separable Data")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

# (b) Cross-validation with different C values
C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
cv_errors = []
train_errors = []
misclassified_counts = []

for C in C_values:
    svm = SVC(kernel='linear', C=C)

    # Compute cross-validation accuracy (5-fold)
    cv_scores = cross_val_score(svm, X_train, y_train, cv=5)
    cv_errors.append(1 - np.mean(cv_scores))

    # Fit on full training data and compute training error
    svm.fit(X_train, y_train)
    train_pred = svm.predict(X_train)
    train_errors.append(1 - accuracy_score(y_train, train_pred))

    # Count misclassified training observations
    misclassified_counts.append(np.sum(y_train != train_pred))

# Plot results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.semilogx(C_values, cv_errors, 'bo-', label='CV Error')
plt.semilogx(C_values, train_errors, 'ro-', label='Train Error')
plt.xlabel("C (Regularization Parameter)")
plt.ylabel("Error Rate")
plt.title("(b) Error Rates vs C")
plt.legend()

plt.subplot(1, 2, 2)
plt.semilogx(C_values, misclassified_counts, 'go-')
plt.xlabel("C (Regularization Parameter)")
plt.ylabel("Number Misclassified")
plt.title("Training Observations Misclassified")
plt.tight_layout()
plt.show()

# (c) Generate test data and evaluate
X_test, y_test = generate_barely_separable_data(1000)
test_errors = []

for C in C_values:
    svm = SVC(kernel='linear', C=C)
    svm.fit(X_train, y_train)
    test_pred = svm.predict(X_test)
    test_errors.append(1 - accuracy_score(y_test, test_pred))

# Find optimal C values
optimal_c_cv = C_values[np.argmin(cv_errors)]
optimal_c_test = C_values[np.argmin(test_errors)]
optimal_c_train = C_values[np.argmin(train_errors)]

# Plot test errors
plt.figure(figsize=(8, 5))
plt.semilogx(C_values, test_errors, 'mo-')
plt.xlabel("C (Regularization Parameter)")
plt.ylabel("Test Error Rate")
plt.title("(c) Test Error Rates vs C")
plt.axvline(x=optimal_c_test, color='k', linestyle='--',
            label=f'Best C (test) = {optimal_c_test}')
plt.legend()
plt.show()

# (d) Print results summary
print("\nResults Summary:")
print(
    f"Best C for training error: {optimal_c_train} (misclassified: {misclassified_counts[C_values.index(optimal_c_train)]})")
print(f"Best C for CV error: {optimal_c_cv} (misclassified: {misclassified_counts[C_values.index(optimal_c_cv)]})")
print(
    f"Best C for test error: {optimal_c_test} (misclassified: {misclassified_counts[C_values.index(optimal_c_test)]})")