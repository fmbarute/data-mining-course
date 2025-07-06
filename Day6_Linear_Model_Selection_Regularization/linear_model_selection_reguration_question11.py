import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from itertools import combinations
from tqdm import tqdm

# Load Boston dataset
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = boston.target  # per capita crime rate

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# (a) Try different regression methods

# 1. Linear Regression (baseline)
lm = LinearRegression()
lm_scores = cross_val_score(lm, X_train_scaled, y_train,
                          cv=5, scoring='neg_mean_squared_error')
lm_mse = -lm_scores.mean()
lm.fit(X_train_scaled, y_train)
lm_test_mse = mean_squared_error(y_test, lm.predict(X_test_scaled))

# 2. Ridge Regression
alphas = np.logspace(-4, 4, 100)
ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X_train_scaled, y_train)
ridge_mse = mean_squared_error(y_train, ridge_cv.predict(X_train_scaled))
ridge_test_mse = mean_squared_error(y_test, ridge_cv.predict(X_test_scaled))

# 3. Lasso Regression
lasso_cv = LassoCV(alphas=alphas, cv=5, max_iter=10000, random_state=42)
lasso_cv.fit(X_train_scaled, y_train)
lasso_mse = mean_squared_error(y_train, lasso_cv.predict(X_train_scaled))
lasso_test_mse = mean_squared_error(y_test, lasso_cv.predict(X_test_scaled))
non_zero_coefs = sum(lasso_cv.coef_ != 0)

# 4. Principal Component Regression (PCR)
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
pcr_test_mse = mean_squared_error(y_test, pcr.predict(X_test_scaled))

# 5. Partial Least Squares (PLS)
pls_mses = []

for n_components in range(1, max_components + 1):
    pls = PLSRegression(n_components=n_components)
    scores = cross_val_score(pls, X_train_scaled, y_train,
                           cv=5, scoring='neg_mean_squared_error')
    pls_mses.append(-scores.mean())

best_n_pls = np.argmin(pls_mses) + 1
pls = PLSRegression(n_components=best_n_pls)
pls.fit(X_train_scaled, y_train)
pls_test_mse = mean_squared_error(y_test, pls.predict(X_test_scaled))

# Results comparison
results = pd.DataFrame({
    'Method': ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'PCR', 'PLS'],
    'CV MSE': [lm_mse, ridge_mse, lasso_mse, pcr_mses[best_n_pcr-1], pls_mses[best_n_pls-1]],
    'Test MSE': [lm_test_mse, ridge_test_mse, lasso_test_mse, pcr_test_mse, pls_test_mse],
    'Parameters': ['-',
                 f'α={ridge_cv.alpha_:.4f}',
                 f'α={lasso_cv.alpha_:.4f}, {non_zero_coefs} features',
                 f'M={best_n_pcr}',
                 f'M={best_n_pls}']
})

print("(a) Regression Method Comparison:")
print(results.sort_values('Test MSE'))

# (b) Propose best model(s)
print("\n(b) Recommended Model:")
print("Based on test MSE, the best performing model is:",
      results.loc[results['Test MSE'].idxmin(), 'Method'])

# (c) Feature usage in best model
if results.loc[results['Test MSE'].idxmin(), 'Method'] == 'Lasso Regression':
    selected_features = X.columns[lasso_cv.coef_ != 0]
    print("\n(c) Selected Features in Lasso Model:")
    print(selected_features)
    print("\nReason: Lasso automatically performs feature selection by shrinking some coefficients to zero.")
    print("This simplifies the model while maintaining good predictive performance.")
else:
    print("\n(c) The best model uses all features, but with regularization (Ridge) or")
    print("dimensionality reduction (PCR/PLS) to handle multicollinearity and overfitting.")

# Plot feature coefficients for interpretation
plt.figure(figsize=(12, 6))
plt.bar(X.columns, lasso_cv.coef_)
plt.xticks(rotation=90)
plt.ylabel('Coefficient Value')
plt.title('Lasso Regression Coefficients')
plt.axhline(0, color='black', linewidth=0.5)
plt.tight_layout()
plt.show()