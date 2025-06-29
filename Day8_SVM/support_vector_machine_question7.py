import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from ISLP import load_data

# Load Auto dataset
auto = load_data('Auto')

# (a) Create binary variable for gas mileage
median_mpg = np.median(auto['mpg'])
auto['high_mpg'] = (auto['mpg'] > median_mpg).astype(int)

# Prepare features (excluding mpg and the target variable)
X = auto.drop(['mpg', 'high_mpg'], axis=1)  # Removed 'name' from drop list
y = auto['high_mpg']

# Convert categorical variables to dummy variables (origin is categorical)
X = pd.get_dummies(X, columns=['origin'], drop_first=True)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# (b) Linear SVM with different C values
C_values = [0.001, 0.01, 0.1, 1, 10, 100]
linear_cv_scores = []

for C in C_values:
    svm_linear = SVC(kernel='linear', C=C)
    scores = cross_val_score(svm_linear, X_scaled, y, cv=5)
    linear_cv_scores.append(np.mean(scores))

# Plot results for linear SVM
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.semilogx(C_values, linear_cv_scores, 'bo-')
plt.xlabel('C (Regularization Parameter)')
plt.ylabel('Cross-Validation Accuracy')
plt.title('(b) Linear SVM Performance')

# (c) Non-linear SVMs with different kernels
# Radial (RBF) kernel with different gamma and C
rbf_params = {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 10]}
rbf_cv_scores = []

for C in rbf_params['C']:
    for gamma in rbf_params['gamma']:
        svm_rbf = SVC(kernel='rbf', C=C, gamma=gamma)
        scores = cross_val_score(svm_rbf, X_scaled, y, cv=5)
        rbf_cv_scores.append((C, gamma, np.mean(scores)))

# Polynomial kernel with different degree and C
poly_params = {'C': [0.1, 1, 10], 'degree': [2, 3]}
poly_cv_scores = []

for C in poly_params['C']:
    for degree in poly_params['degree']:
        svm_poly = SVC(kernel='poly', C=C, degree=degree)
        scores = cross_val_score(svm_poly, X_scaled, y, cv=5)
        poly_cv_scores.append((C, degree, np.mean(scores)))

# Display results for non-linear SVMs
rbf_results = pd.DataFrame(rbf_cv_scores, columns=['C', 'gamma', 'CV Accuracy'])
poly_results = pd.DataFrame(poly_cv_scores, columns=['C', 'degree', 'CV Accuracy'])

print("\nRBF Kernel Results:")
print(rbf_results.sort_values('CV Accuracy', ascending=False).head())

print("\nPolynomial Kernel Results:")
print(poly_results.sort_values('CV Accuracy', ascending=False).head())

# Plot best RBF and Polynomial models
best_rbf = rbf_results.loc[rbf_results['CV Accuracy'].idxmax()]
best_poly = poly_results.loc[poly_results['CV Accuracy'].idxmax()]

plt.subplot(1, 2, 2)
plt.bar(['Linear', 'RBF', 'Polynomial'],
        [max(linear_cv_scores), best_rbf['CV Accuracy'], best_poly['CV Accuracy']])
plt.ylabel('Best CV Accuracy')
plt.title('(c) Best Performance by Kernel Type')
plt.tight_layout()
plt.show()

# (d) Feature importance analysis (for linear SVM)
svm_linear = SVC(kernel='linear', C=1)
svm_linear.fit(X_scaled, y)

# Get feature importances (absolute coefficients)
coef = svm_linear.coef_[0]
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': np.abs(coef)
}).sort_values('Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Importance (absolute coefficient value)')
plt.title('(d) Feature Importance from Linear SVM')
plt.gca().invert_yaxis()
plt.show()