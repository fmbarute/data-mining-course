import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Generate simulated non-linear data (moons pattern)
X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

# 2. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Create SVM models with different kernels
models = {
    "Linear SVM": SVC(kernel="linear", C=1),
    "Polynomial SVM (deg=3)": SVC(kernel="poly", degree=3, C=1),
    "RBF SVM": SVC(kernel="rbf", gamma=1, C=1)
}

# 4. Train models and evaluate performance
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    results[name] = {"Train Accuracy": train_acc, "Test Accuracy": test_acc}

# 5. Print results
print("{:<20} {:<15} {:<15}".format("Model", "Train Accuracy", "Test Accuracy"))
for name, res in results.items():
    print("{:<20} {:<15.4f} {:<15.4f}".format(name, res["Train Accuracy"], res["Test Accuracy"]))


# 6. Create visualization
def plot_decision_boundary(model, X, y, ax, title):
    # Create mesh
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict on mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot
    ax.contourf(xx, yy, Z, alpha=0.3)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    ax.set_title(title)


# 7. Plot decision boundaries
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for (name, model), ax in zip(models.items(), axes):
    plot_decision_boundary(model, X_train, y_train, ax, name)
plt.tight_layout()
plt.show()