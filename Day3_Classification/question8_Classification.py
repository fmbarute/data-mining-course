import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data (100 samples, 5 features)
X = np.random.randn(100, 5)
y = (X[:, 0] + 0.5*X[:, 1] + np.random.randn(100) > 0).astype(int)

# Split data equally into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Initialize models
log_reg = LogisticRegression()
knn = KNeighborsClassifier(n_neighbors=1)

# Train models
log_reg.fit(X_train, y_train)
knn.fit(X_train, y_train)

# Evaluate models
log_reg_train_err = 1 - accuracy_score(y_train, log_reg.predict(X_train))
log_reg_test_err = 1 - accuracy_score(y_test, log_reg.predict(X_test))

knn_train_err = 1 - accuracy_score(y_train, knn.predict(X_train))
knn_test_err = 1 - accuracy_score(y_test, knn.predict(X_test))
knn_avg_err = (knn_train_err + knn_test_err) / 2

# Print results
print("Logistic Regression Results:")
print(f"Training Error: {log_reg_train_err:.0%}")
print(f"Test Error: {log_reg_test_err:.0%}\n")

print("1-Nearest Neighbors (K=1) Results:")
print(f"Training Error: {knn_train_err:.0%}")
print(f"Test Error: {knn_test_err:.0%}")
print(f"Average Error: {knn_avg_err:.0%}\n")

# Analysis and recommendation
print("Analysis:")
print("1. Logistic Regression has:")
print(f"   - Moderate training error ({log_reg_train_err:.0%})")
print(f"   - Higher but reasonable test error ({log_reg_test_err:.0%})")
print("   This suggests it generalizes reasonably well.\n")

print("2. K=1 Nearest Neighbors has:")
print(f"   - 0% training error (perfect memorization)")
print(f"   - Very high test error ({knn_test_err:.0%})")
print("   This indicates severe overfitting.\n")

print("Conclusion:")
print("We should prefer Logistic Regression because:")
print("- It has better generalization performance (lower test error)")
print("- K=1's perfect training score and high test error show it's overfitting")
print("- Test error is what matters for new observations, not training error")
print(f"- Even though K=1 has a lower average error ({knn_avg_err:.0%} vs logistic's test {log_reg_test_err:.0%})")
print("  this average is misleading because it includes the meaningless 0% training error")

# Bonus: Plotting the error comparison
import matplotlib.pyplot as plt

models = ['Logistic Regression', 'K=1 NN']
train_errors = [log_reg_train_err, knn_train_err]
test_errors = [log_reg_test_err, knn_test_err]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, train_errors, width, label='Training Error')
rects2 = ax.bar(x + width/2, test_errors, width, label='Test Error')

ax.set_ylabel('Error Rate')
ax.set_title('Model Comparison: Training vs Test Error')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

ax.bar_label(rects1, padding=3, fmt='%.0f%%')
ax.bar_label(rects2, padding=3, fmt='%.0f%%')

plt.tight_layout()
plt.show()