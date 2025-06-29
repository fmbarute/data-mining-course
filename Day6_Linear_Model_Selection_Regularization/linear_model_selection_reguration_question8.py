import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import statsmodels.api as sm
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)

# (a) Generate predictor X and noise vector ϵ
n = 100
X = np.random.normal(size=n)
epsilon = np.random.normal(scale=0.5, size=n)  # Reduced noise for better convergence

# (b) Generate response Y with cubic relationship
beta0, beta1, beta2, beta3 = 2, 1, -0.5, 0.1
Y = beta0 + beta1 * X + beta2 * (X ** 2) + beta3 * (X ** 3) + epsilon

# Create polynomial features up to X^10 and standardize
poly = PolynomialFeatures(degree=10, include_bias=False)
X_poly = poly.fit_transform(X.reshape(-1, 1))
scaler = StandardScaler()
X_poly_scaled = scaler.fit_transform(X_poly)  # Scaling helps with convergence
feature_names = [f'X^{i + 1}' for i in range(10)]


# (c) Forward stepwise selection with Cp (equivalent to AIC in statsmodels)
def forward_stepwise(X, y, feature_names):
    included = []
    best_aic = np.inf
    best_model = None
    results = []

    while len(included) < X.shape[1]:
        remaining = [f for f in range(X.shape[1]) if f not in included]
        aic_candidates = []

        for new in remaining:
            current = included + [new]
            X_current = X[:, current]
            model = OLS(y, add_constant(X_current)).fit()
            aic_candidates.append((new, model.aic))

        # Select the feature that gives lowest AIC
        new, aic = min(aic_candidates, key=lambda x: x[1])

        if aic < best_aic:
            included.append(new)
            best_aic = aic
            best_model = OLS(y, add_constant(X[:, included])).fit()
            results.append({
                'num_features': len(included),
                'features': [feature_names[i] for i in included],
                'AIC': aic,
                'coefficients': best_model.params[1:],  # exclude intercept
                'intercept': best_model.params[0]
            })
        else:
            break

    return results


forward_results = forward_stepwise(X_poly_scaled, Y, feature_names)
best_forward = forward_results[-1]

print("\n(c) Forward Stepwise Results:")
print(f"Selected features: {best_forward['features']}")
print(f"Intercept: {best_forward['intercept']:.4f}")
for f, coef in zip(best_forward['features'], best_forward['coefficients']):
    print(f"{f}: {coef:.4f}")


# (d) Backward stepwise selection
def backward_stepwise(X, y, feature_names):
    included = list(range(X.shape[1]))
    current_aic = OLS(y, add_constant(X)).fit().aic
    results = []

    while len(included) > 1:
        aic_candidates = []

        for remove in included:
            current = [f for f in included if f != remove]
            X_current = X[:, current]
            model = OLS(y, add_constant(X_current)).fit()
            aic_candidates.append((remove, model.aic))

        # Select the removal that gives lowest AIC
        remove, aic = min(aic_candidates, key=lambda x: x[1])

        if aic < current_aic:
            included.remove(remove)
            current_aic = aic
            best_model = OLS(y, add_constant(X[:, included])).fit()
            results.append({
                'num_features': len(included),
                'features': [feature_names[i] for i in included],
                'AIC': aic,
                'coefficients': best_model.params[1:],
                'intercept': best_model.params[0]
            })
        else:
            break

    return results


backward_results = backward_stepwise(X_poly_scaled, Y, feature_names)
best_backward = backward_results[-1]

print("\n(d) Backward Stepwise Results:")
print(f"Selected features: {best_backward['features']}")
print(f"Intercept: {best_backward['intercept']:.4f}")
for f, coef in zip(best_backward['features'], best_backward['coefficients']):
    print(f"{f}: {coef:.4f}")

# (e) Lasso with cross-validation - with increased max_iter and better alpha range
lasso = LassoCV(cv=5,
                alphas=np.logspace(-6, 2, 100),  # Wider range of alphas
                max_iter=10000,  # Increased iterations
                tol=1e-4,  # Looser tolerance
                random_state=42)
lasso.fit(X_poly_scaled, Y)

# Plot cross-validation error
plt.figure(figsize=(10, 6))
plt.semilogx(lasso.alphas_, lasso.mse_path_.mean(axis=1), 'b-')
plt.axvline(lasso.alpha_, color='red', linestyle='--')
plt.xlabel('Lambda (α)')
plt.ylabel('Mean squared error')
plt.title('Lasso Cross-Validation Error')
plt.show()

print("\n(e) Lasso Results:")
print(f"Optimal lambda: {lasso.alpha_:.4f}")
print(f"Intercept: {lasso.intercept_:.4f}")
for i, coef in enumerate(lasso.coef_):
    if abs(coef) > 1e-4:  # Only show non-zero coefficients
        print(f"X^{i + 1}: {coef:.4f}")

# (f) New response with only X^7 term
beta0_new, beta7 = 1, 2
Y_new = beta0_new + beta7 * (X ** 7) + np.random.normal(scale=0.5, size=n)

# Forward stepwise on new data
forward_results_new = forward_stepwise(X_poly_scaled, Y_new, feature_names)
best_forward_new = forward_results_new[-1]

print("\n(f) Forward Stepwise with X^7 only model:")
print(f"Selected features: {best_forward_new['features']}")
print(f"Intercept: {best_forward_new['intercept']:.4f}")
for f, coef in zip(best_forward_new['features'], best_forward_new['coefficients']):
    print(f"{f}: {coef:.4f}")

# Lasso on new data with increased iterations
lasso_new = LassoCV(cv=5,
                    alphas=np.logspace(-6, 2, 100),
                    max_iter=10000,
                    tol=1e-4,
                    random_state=42)
lasso_new.fit(X_poly_scaled, Y_new)

print("\nLasso with X^7 only model:")
print(f"Optimal lambda: {lasso_new.alpha_:.4f}")
print(f"Intercept: {lasso_new.intercept_:.4f}")
for i, coef in enumerate(lasso_new.coef_):
    if abs(coef) > 1e-4:
        print(f"X^{i + 1}: {coef:.4f}")