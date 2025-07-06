import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from ISLP import load_data

# Load OJ dataset
oj = load_data('OJ')

# (a) Create training (800 obs) and test sets
X = oj.drop('Purchase', axis=1)
y = oj['Purchase']

# Convert categorical variables to numeric
le = LabelEncoder()
y = le.fit_transform(y)  # CH=1, MM=0
X = pd.get_dummies(X, drop_first=True)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data (800 train, remaining test)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, train_size=800, random_state=42)

# (b) Linear SVC with C=0.01
svc_001 = SVC(kernel='linear', C=0.01)
svc_001.fit(X_train, y_train)

# Number of support vectors
n_support = svc_001.n_support_
print(f"(b) Number of support vectors (class 0, class 1): {n_support}")
print(f"    Total support points: {sum(n_support)}")

# (c) Training and test error rates
train_pred = svc_001.predict(X_train)
test_pred = svc_001.predict(X_test)
print(f"(c) Training error: {1 - accuracy_score(y_train, train_pred):.4f}")
print(f"    Test error: {1 - accuracy_score(y_test, test_pred):.4f}")

# (d) Cross-validation for optimal C
param_grid = {'C': np.logspace(-2, 1, 20)}  # 0.01 to 10
grid_search = GridSearchCV(SVC(kernel='linear'), param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_C = grid_search.best_params_['C']
print(f"(d) Optimal C from CV: {best_C:.4f}")

# (e) Error rates with optimal C
svc_optimal = SVC(kernel='linear', C=best_C)
svc_optimal.fit(X_train, y_train)
train_pred_opt = svc_optimal.predict(X_train)
test_pred_opt = svc_optimal.predict(X_test)
print(f"(e) Training error (optimal C): {1 - accuracy_score(y_train, train_pred_opt):.4f}")
print(f"    Test error (optimal C): {1 - accuracy_score(y_test, test_pred_opt):.4f}")

# (f) Repeat with RBF kernel
# (f-b) RBF SVC with C=0.01
svm_rbf_001 = SVC(kernel='rbf', C=0.01)
svm_rbf_001.fit(X_train, y_train)
print("\n(f) RBF Kernel Results:")
print(f"Support vectors (class 0, class 1): {svm_rbf_001.n_support_}")
print(f"Total support points: {sum(svm_rbf_001.n_support_)}")

# (f-c) Error rates
rbf_train_pred = svm_rbf_001.predict(X_train)
rbf_test_pred = svm_rbf_001.predict(X_test)
print(f"Training error: {1 - accuracy_score(y_train, rbf_train_pred):.4f}")
print(f"Test error: {1 - accuracy_score(y_test, rbf_test_pred):.4f}")

# (f-d) CV for optimal C with RBF
rbf_grid = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
rbf_grid.fit(X_train, y_train)
best_C_rbf = rbf_grid.best_params_['C']
print(f"Optimal C (RBF): {best_C_rbf:.4f}")

# (f-e) Error rates with optimal RBF
svm_rbf_opt = SVC(kernel='rbf', C=best_C_rbf)
svm_rbf_opt.fit(X_train, y_train)
rbf_train_opt = svm_rbf_opt.predict(X_train)
rbf_test_opt = svm_rbf_opt.predict(X_test)
print(f"Training error (optimal): {1 - accuracy_score(y_train, rbf_train_opt):.4f}")
print(f"Test error (optimal): {1 - accuracy_score(y_test, rbf_test_opt):.4f}")

# (g) Repeat with polynomial kernel (degree=2)
# (g-b) Poly SVC with C=0.01
svm_poly_001 = SVC(kernel='poly', degree=2, C=0.01)
svm_poly_001.fit(X_train, y_train)
print("\n(g) Polynomial Kernel (degree=2) Results:")
print(f"Support vectors (class 0, class 1): {svm_poly_001.n_support_}")
print(f"Total support points: {sum(svm_poly_001.n_support_)}")

# (g-c) Error rates
poly_train_pred = svm_poly_001.predict(X_train)
poly_test_pred = svm_poly_001.predict(X_test)
print(f"Training error: {1 - accuracy_score(y_train, poly_train_pred):.4f}")
print(f"Test error: {1 - accuracy_score(y_test, poly_test_pred):.4f}")

# (g-d) CV for optimal C with poly
poly_grid = GridSearchCV(SVC(kernel='poly', degree=2), param_grid, cv=5)
poly_grid.fit(X_train, y_train)
best_C_poly = poly_grid.best_params_['C']
print(f"Optimal C (poly): {best_C_poly:.4f}")

# (g-e) Error rates with optimal poly
svm_poly_opt = SVC(kernel='poly', degree=2, C=best_C_poly)
svm_poly_opt.fit(X_train, y_train)
poly_train_opt = svm_poly_opt.predict(X_train)
poly_test_opt = svm_poly_opt.predict(X_test)
print(f"Training error (optimal): {1 - accuracy_score(y_train, poly_train_opt):.4f}")
print(f"Test error (optimal): {1 - accuracy_score(y_test, poly_test_opt):.4f}")

# (h) Compare all approaches
results = {
    'Linear SVM (C=0.01)': 1 - accuracy_score(y_test, test_pred),
    'Linear SVM (optimal C)': 1 - accuracy_score(y_test, test_pred_opt),
    'RBF SVM (C=0.01)': 1 - accuracy_score(y_test, rbf_test_pred),
    'RBF SVM (optimal C)': 1 - accuracy_score(y_test, rbf_test_opt),
    'Poly SVM (C=0.01)': 1 - accuracy_score(y_test, poly_test_pred),
    'Poly SVM (optimal C)': 1 - accuracy_score(y_test, poly_test_opt)
}

print("\n(h) Test Error Comparison:")
for model, error in sorted(results.items(), key=lambda x: x[1]):
    print(f"{model}: {error:.4f}")