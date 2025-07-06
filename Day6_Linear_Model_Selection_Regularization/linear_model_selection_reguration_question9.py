import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the College dataset
# Note: You may need to adjust the path to your dataset
college = pd.read_csv('College.csv')
college = college.drop('Unnamed: 0', axis=1)  # Remove the first column (college names)

# Convert categorical variables to dummy variables
college = pd.get_dummies(college, columns=['Private'], drop_first=True)

# (a) Split into training and test sets
X = college.drop('Apps', axis=1)
y = college['Apps']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features (important for regularized methods and PCR/PLS)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# (b) Linear regression (least squares)
lm = LinearRegression()
lm.fit(X_train_scaled, y_train)
lm_pred = lm.predict(X_test_scaled)
lm_mse = mean_squared_error(y_test, lm_pred)
print(f"\n(b) Linear Regression Test MSE: {lm_mse:.2f}")

# (c) Ridge regression with cross-validation
alphas = np.logspace(-4, 4, 100)  # Range of alpha values to try
ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X_train_scaled, y_train)
ridge_pred = ridge_cv.predict(X_test_scaled)
ridge_mse = mean_squared_error(y_test, ridge_pred)
print(f"(c) Ridge Regression (alpha={ridge_cv.alpha_:.4f}) Test MSE: {ridge_mse:.2f}")

# (d) Lasso with cross-validation
lasso_cv = LassoCV(alphas=alphas, cv=5, max_iter=10000, random_state=42)
lasso_cv.fit(X_train_scaled, y_train)
lasso_pred = lasso_cv.predict(X_test_scaled)
lasso_mse = mean_squared_error(y_test, lasso_pred)
print(f"(d) Lasso Regression (alpha={lasso_cv.alpha_:.4f}) Test MSE: {lasso_mse:.2f}")
print(f"    Number of non-zero coefficients: {sum(lasso_cv.coef_ != 0)}")

# (e) Principal Component Regression (PCR) with CV
# We'll use a pipeline with PCA and linear regression
max_components = min(X_train_scaled.shape[0], X_train_scaled.shape[1])
pcr_mses = []

for n_components in range(1, max_components + 1):
    pcr = make_pipeline(PCA(n_components=n_components), LinearRegression())
    scores = cross_val_score(pcr, X_train_scaled, y_train,
                           cv=5, scoring='neg_mean_squared_error')
    pcr_mses.append(-scores.mean())

best_n_pcr = np.argmin(pcr_mses) + 1
pcr = make_pipeline(PCA(n_components=best_n_pcr), LinearRegression())
pcr.fit(X_train_scaled, y_train)
pcr_pred = pcr.predict(X_test_scaled)
pcr_mse = mean_squared_error(y_test, pcr_pred)
print(f"(e) PCR (M={best_n_pcr}) Test MSE: {pcr_mse:.2f}")

# Plot PCR MSE vs number of components
plt.figure(figsize=(10, 6))
plt.plot(range(1, max_components + 1), pcr_mses, '-o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cross-Validated MSE')
plt.title('PCR: MSE vs Number of Components')
plt.axvline(best_n_pcr, color='red', linestyle='--')
plt.show()

# (f) Partial Least Squares (PLS) with CV
pls_mses = []

for n_components in range(1, max_components + 1):
    pls = PLSRegression(n_components=n_components)
    scores = cross_val_score(pls, X_train_scaled, y_train,
                           cv=5, scoring='neg_mean_squared_error')
    pls_mses.append(-scores.mean())

best_n_pls = np.argmin(pls_mses) + 1
pls = PLSRegression(n_components=best_n_pls)
pls.fit(X_train_scaled, y_train)
pls_pred = pls.predict(X_test_scaled)
pls_mse = mean_squared_error(y_test, pls_pred)
print(f"(f) PLS (M={best_n_pls}) Test MSE: {pls_mse:.2f}")

# Plot PLS MSE vs number of components
plt.figure(figsize=(10, 6))
plt.plot(range(1, max_components + 1), pls_mses, '-o')
plt.xlabel('Number of Components')
plt.ylabel('Cross-Validated MSE')
plt.title('PLS: MSE vs Number of Components')
plt.axvline(best_n_pls, color='red', linestyle='--')
plt.show()

# (g) Compare results
results = pd.DataFrame({
    'Method': ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'PCR', 'PLS'],
    'Test MSE': [lm_mse, ridge_mse, lasso_mse, pcr_mse, pls_mse],
    'Parameters': ['-', f'α={ridge_cv.alpha_:.4f}',
                  f'α={lasso_cv.alpha_:.4f}, {sum(lasso_cv.coef_ != 0)} non-zero',
                  f'M={best_n_pcr}', f'M={best_n_pls}']
})

print("\n(g) Comparison of Results:")
print(results.sort_values('Test MSE'))