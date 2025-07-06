#!/usr/bin/env python3
"""
Exercise 8: Forward and Backward Stepwise Selection with Lasso Regression
Complete implementation with detailed analysis and visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table
import warnings

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Configure plotting
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)


class StepwiseRegression:
    """Implementation of forward and backward stepwise selection"""

    def __init__(self, method='cp'):
        """
        Initialize stepwise regression
        method: 'cp' for Mallows Cp, 'aic' for AIC, 'bic' for BIC
        """
        self.method = method
        self.results = []

    def mallows_cp(self, y_true, y_pred, p, sigma2_full):
        """Calculate Mallows' Cp statistic"""
        n = len(y_true)
        mse = mean_squared_error(y_true, y_pred)
        cp = (mse / sigma2_full) * n - n + 2 * p
        return cp

    def aic(self, y_true, y_pred, p):
        """Calculate AIC"""
        n = len(y_true)
        mse = mean_squared_error(y_true, y_pred)
        return n * np.log(mse) + 2 * p

    def bic(self, y_true, y_pred, p):
        """Calculate BIC"""
        n = len(y_true)
        mse = mean_squared_error(y_true, y_pred)
        return n * np.log(mse) + p * np.log(n)

    def forward_selection(self, X, y, feature_names=None):
        """Forward stepwise selection"""
        n, p = X.shape
        if feature_names is None:
            feature_names = [f'X{i + 1}' for i in range(p)]

        # Fit full model for Cp calculation
        X_full = sm.add_constant(X)
        full_model = sm.OLS(y, X_full).fit()
        sigma2_full = full_model.mse_resid

        # Initialize
        selected = []
        remaining = list(range(p))
        results = []

        print("FORWARD STEPWISE SELECTION")
        print("=" * 80)
        print(f"{'Step':<4} {'Added':<10} {'Cp':<10} {'AIC':<10} {'BIC':<10} {'Selected Features'}")
        print("-" * 80)

        for step in range(p):
            best_score = float('inf')
            best_feature = None
            best_stats = {}

            # Try adding each remaining feature
            for feature in remaining:
                trial_features = selected + [feature]
                X_trial = sm.add_constant(X[:, trial_features])

                try:
                    model = sm.OLS(y, X_trial).fit()
                    y_pred = model.fittedvalues
                    p_trial = len(trial_features) + 1  # +1 for intercept

                    # Calculate criteria
                    cp = self.mallows_cp(y, y_pred, p_trial, sigma2_full)
                    aic = self.aic(y, y_pred, p_trial)
                    bic = self.bic(y, y_pred, p_trial)

                    # Select based on chosen method
                    if self.method == 'cp':
                        score = cp
                    elif self.method == 'aic':
                        score = aic
                    else:  # bic
                        score = bic

                    if score < best_score:
                        best_score = score
                        best_feature = feature
                        best_stats = {'cp': cp, 'aic': aic, 'bic': bic}

                except np.linalg.LinAlgError:
                    continue

            if best_feature is not None:
                selected.append(best_feature)
                remaining.remove(best_feature)

                selected_names = [feature_names[i] for i in selected]
                results.append({
                    'step': step + 1,
                    'added': feature_names[best_feature],
                    'selected_indices': selected.copy(),
                    'selected_names': selected_names.copy(),
                    'cp': best_stats['cp'],
                    'aic': best_stats['aic'],
                    'bic': best_stats['bic']
                })

                print(f"{step + 1:<4} {feature_names[best_feature]:<10} "
                      f"{best_stats['cp']:<10.2f} {best_stats['aic']:<10.2f} "
                      f"{best_stats['bic']:<10.2f} {', '.join(selected_names)}")
            else:
                break

        # Find best model
        best_model = min(results, key=lambda x: x[self.method])
        print(f"\nBest model (Step {best_model['step']}, {self.method.upper()} = {best_model[self.method]:.2f}):")
        print(f"Features: {', '.join(best_model['selected_names'])}")

        return results, best_model

    def backward_selection(self, X, y, feature_names=None):
        """Backward stepwise selection"""
        n, p = X.shape
        if feature_names is None:
            feature_names = [f'X{i + 1}' for i in range(p)]

        # Fit full model for Cp calculation
        X_full = sm.add_constant(X)
        full_model = sm.OLS(y, X_full).fit()
        sigma2_full = full_model.mse_resid

        # Start with all features
        selected = list(range(p))
        results = []

        print("BACKWARD STEPWISE SELECTION")
        print("=" * 80)
        print(f"{'Step':<4} {'Removed':<10} {'Cp':<10} {'AIC':<10} {'BIC':<10} {'Remaining Features'}")
        print("-" * 80)

        # Initial model (full model)
        y_pred = full_model.fittedvalues
        p_full = p + 1  # +1 for intercept
        cp_init = self.mallows_cp(y, y_pred, p_full, sigma2_full)
        aic_init = self.aic(y, y_pred, p_full)
        bic_init = self.bic(y, y_pred, p_full)

        selected_names = [feature_names[i] for i in selected]
        results.append({
            'step': 0,
            'removed': 'None',
            'selected_indices': selected.copy(),
            'selected_names': selected_names.copy(),
            'cp': cp_init,
            'aic': aic_init,
            'bic': bic_init
        })

        print(f"{0:<4} {'None':<10} {cp_init:<10.2f} {aic_init:<10.2f} "
              f"{bic_init:<10.2f} {', '.join(selected_names)}")

        # Remove features one by one
        for step in range(p):
            if len(selected) <= 1:
                break

            best_score = float('inf')
            worst_feature = None
            best_stats = {}

            # Try removing each feature
            for feature in selected:
                trial_features = [f for f in selected if f != feature]

                if len(trial_features) == 0:
                    continue

                X_trial = sm.add_constant(X[:, trial_features])

                try:
                    model = sm.OLS(y, X_trial).fit()
                    y_pred = model.fittedvalues
                    p_trial = len(trial_features) + 1  # +1 for intercept

                    # Calculate criteria
                    cp = self.mallows_cp(y, y_pred, p_trial, sigma2_full)
                    aic = self.aic(y, y_pred, p_trial)
                    bic = self.bic(y, y_pred, p_trial)

                    # Select based on chosen method
                    if self.method == 'cp':
                        score = cp
                    elif self.method == 'aic':
                        score = aic
                    else:  # bic
                        score = bic

                    if score < best_score:
                        best_score = score
                        worst_feature = feature
                        best_stats = {'cp': cp, 'aic': aic, 'bic': bic}

                except np.linalg.LinAlgError:
                    continue

            if worst_feature is not None:
                selected.remove(worst_feature)
                selected_names = [feature_names[i] for i in selected]

                results.append({
                    'step': step + 1,
                    'removed': feature_names[worst_feature],
                    'selected_indices': selected.copy(),
                    'selected_names': selected_names.copy(),
                    'cp': best_stats['cp'],
                    'aic': best_stats['aic'],
                    'bic': best_stats['bic']
                })

                remaining_str = ', '.join(selected_names) if selected_names else 'None'
                print(f"{step + 1:<4} {feature_names[worst_feature]:<10} "
                      f"{best_stats['cp']:<10.2f} {best_stats['aic']:<10.2f} "
                      f"{best_stats['bic']:<10.2f} {remaining_str}")
            else:
                break

        # Find best model
        best_model = min(results, key=lambda x: x[self.method])
        print(f"\nBest model (Step {best_model['step']}, {self.method.upper()} = {best_model[self.method]:.2f}):")
        if best_model['selected_names']:
            print(f"Features: {', '.join(best_model['selected_names'])}")
        else:
            print("Features: Intercept only")

        return results, best_model


def print_section_header(title):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(f"{title}")
    print("=" * 70)


def fit_and_report_model(X, y, selected_indices, feature_names, title="Model"):
    """Fit final model and report detailed results"""
    if len(selected_indices) == 0:
        print(f"\n{title}: No features selected!")
        return None

    # Fit the model
    X_selected = X[:, selected_indices]
    X_with_const = sm.add_constant(X_selected)
    model = sm.OLS(y, X_with_const).fit()

    print(f"\n{title} Results:")
    print("-" * 50)
    print(f"Selected features: {', '.join([feature_names[i] for i in selected_indices])}")
    print(f"R-squared: {model.rsquared:.4f}")
    print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
    print(f"MSE: {model.mse_resid:.4f}")
    print(f"AIC: {model.aic:.2f}")
    print(f"BIC: {model.bic:.2f}")

    print(f"\nCoefficients:")
    print(f"Intercept: {model.params[0]:.4f} (p-value: {model.pvalues[0]:.4f})")
    for i, idx in enumerate(selected_indices):
        coef = model.params[i + 1]
        pval = model.pvalues[i + 1]
        print(f"{feature_names[idx]}: {coef:.4f} (p-value: {pval:.4f})")

    return model


# PART (A) and (B): Data Generation
print_section_header("PARTS (A) & (B): DATA GENERATION")

# Generate predictor X and noise ε
n = 100
X = np.random.normal(0, 1, n)
epsilon = np.random.normal(0, 1, n)

# Generate response Y with chosen coefficients
beta0, beta1, beta2, beta3 = 3.0, 2.0, -1.5, 0.5
Y = beta0 + beta1 * X + beta2 * (X ** 2) + beta3 * (X ** 3) + epsilon

print(f"Sample size: n = {n}")
print(f"True model: Y = {beta0} + {beta1}*X + {beta2}*X² + {beta3}*X³ + ε")
print(f"X statistics: mean = {X.mean():.3f}, std = {X.std():.3f}")
print(f"Y statistics: mean = {Y.mean():.3f}, std = {Y.std():.3f}")

# Create polynomial features X, X², ..., X^10
X_poly = np.column_stack([X ** i for i in range(1, 11)])
feature_names = [f'X^{i}' for i in range(1, 11)]

print(f"Created polynomial features: {', '.join(feature_names)}")

# PART (C): Forward Stepwise Selection
print_section_header("PART (C): FORWARD STEPWISE SELECTION")

forward_selector = StepwiseRegression(method='cp')
forward_results, best_forward = forward_selector.forward_selection(X_poly, Y, feature_names)

# Fit and report final forward model
forward_model = fit_and_report_model(X_poly, Y, best_forward['selected_indices'],
                                     feature_names, "Forward Selection Final Model")

# PART (D): Backward Stepwise Selection
print_section_header("PART (D): BACKWARD STEPWISE SELECTION")

backward_selector = StepwiseRegression(method='cp')
backward_results, best_backward = backward_selector.backward_selection(X_poly, Y, feature_names)

# Fit and report final backward model
backward_model = fit_and_report_model(X_poly, Y, best_backward['selected_indices'],
                                      feature_names, "Backward Selection Final Model")

# Compare Forward and Backward Results
print_section_header("COMPARISON: FORWARD vs BACKWARD SELECTION")

forward_features = set(best_forward['selected_names'])
backward_features = set(best_backward['selected_names'])

print(f"Forward selection features: {', '.join(sorted(forward_features))}")
print(f"Backward selection features: {', '.join(sorted(backward_features))}")
print(f"Features in common: {', '.join(sorted(forward_features & backward_features))}")
print(f"Forward only: {', '.join(sorted(forward_features - backward_features))}")
print(f"Backward only: {', '.join(sorted(backward_features - forward_features))}")
print(f"Same model selected: {forward_features == backward_features}")

# PART (E): Lasso Regression
print_section_header("PART (E): LASSO REGRESSION")

# Standardize features for lasso
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

# Lasso with cross-validation
alphas = np.logspace(-4, 2, 100)
lasso_cv = LassoCV(alphas=alphas, cv=10, random_state=42, max_iter=2000)
lasso_cv.fit(X_scaled, Y)

print(f"Optimal λ (alpha): {lasso_cv.alpha_:.6f}")
print(f"Cross-validation R²: {lasso_cv.score(X_scaled, Y):.4f}")

# Fit final lasso model
lasso_final = Lasso(alpha=lasso_cv.alpha_, max_iter=2000)
lasso_final.fit(X_scaled, Y)

# Report lasso coefficients
print(f"\nLasso Regression Results:")
print("-" * 50)
print(f"Intercept: {lasso_final.intercept_:.4f}")

selected_lasso = []
for i, (coef, name) in enumerate(zip(lasso_final.coef_, feature_names)):
    print(f"{name}: {coef:.4f}")
    if abs(coef) > 1e-6:
        selected_lasso.append(name)

print(f"\nSelected features (non-zero coefficients): {', '.join(selected_lasso)}")

# Create visualization for lasso
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Cross-validation error
ax1.semilogx(lasso_cv.alphas_, lasso_cv.mse_path_.mean(axis=1), 'b-', alpha=0.8)
ax1.fill_between(lasso_cv.alphas_,
                 lasso_cv.mse_path_.mean(axis=1) - lasso_cv.mse_path_.std(axis=1),
                 lasso_cv.mse_path_.mean(axis=1) + lasso_cv.mse_path_.std(axis=1),
                 alpha=0.3)
ax1.axvline(lasso_cv.alpha_, color='red', linestyle='--',
            label=f'Optimal λ = {lasso_cv.alpha_:.4f}')
ax1.set_xlabel('λ (regularization parameter)')
ax1.set_ylabel('Mean CV Error')
ax1.set_title('Lasso Cross-Validation Error')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Coefficient paths
alphas_path = np.logspace(-4, 2, 50)
coefs_path = []

for alpha in alphas_path:
    lasso_temp = Lasso(alpha=alpha, max_iter=2000)
    lasso_temp.fit(X_scaled, Y)
    coefs_path.append(lasso_temp.coef_)

coefs_path = np.array(coefs_path)

for i in range(len(feature_names)):
    ax2.semilogx(alphas_path, coefs_path[:, i], label=feature_names[i], linewidth=2)

ax2.axvline(lasso_cv.alpha_, color='red', linestyle='--', alpha=0.8)
ax2.set_xlabel('λ (regularization parameter)')
ax2.set_ylabel('Coefficient Value')
ax2.set_title('Lasso Coefficient Paths')
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.grid(True, alpha=0.3)

# Plot 3: Comparison of selected features
methods = ['True Model', 'Forward', 'Backward', 'Lasso']
true_features = ['X^1', 'X^2', 'X^3']

feature_matrix = np.zeros((len(methods), len(feature_names)))
for j, feature in enumerate(feature_names):
    if feature in true_features:
        feature_matrix[0, j] = 1
    if feature in forward_features:
        feature_matrix[1, j] = 1
    if feature in backward_features:
        feature_matrix[2, j] = 1
    if feature in selected_lasso:
        feature_matrix[3, j] = 1

im = ax3.imshow(feature_matrix, cmap='RdYlBu_r', aspect='auto')
ax3.set_xticks(range(len(feature_names)))
ax3.set_xticklabels(feature_names, rotation=45)
ax3.set_yticks(range(len(methods)))
ax3.set_yticklabels(methods)
ax3.set_title('Feature Selection Comparison')

# Add text annotations
for i in range(len(methods)):
    for j in range(len(feature_names)):
        text = ax3.text(j, i, '✓' if feature_matrix[i, j] else '',
                        ha="center", va="center", color="black", fontsize=12)

# Plot 4: Model comparison metrics
cp_scores = [best_forward['cp'], best_backward['cp']]
methods_comp = ['Forward', 'Backward']

ax4.bar(methods_comp, cp_scores, color=['skyblue', 'lightcoral'])
ax4.set_ylabel('Mallows Cp')
ax4.set_title('Model Selection Criteria Comparison')
ax4.grid(True, alpha=0.3)

# Add value labels on bars
for i, v in enumerate(cp_scores):
    ax4.text(i, v + 0.1, f'{v:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# PART (F): Sparse Model Analysis
print_section_header("PART (F): SPARSE MODEL ANALYSIS")

# Generate new response with only X^7 term
np.random.seed(123)  # Different seed for new experiment
epsilon_new = np.random.normal(0, 1, n)
beta0_new, beta7 = 2.0, 0.4
Y_new = beta0_new + beta7 * (X ** 7) + epsilon_new

print(f"New sparse model: Y = {beta0_new} + {beta7}*X^7 + ε")
print(f"Y_new statistics: mean = {Y_new.mean():.3f}, std = {Y_new.std():.3f}")

# Forward stepwise on sparse data
print(f"\nForward Stepwise Selection on Sparse Data:")
forward_sparse = StepwiseRegression(method='cp')
sparse_forward_results, sparse_best_forward = forward_sparse.forward_selection(X_poly, Y_new, feature_names)

# Lasso on sparse data
print(f"\nLasso Regression on Sparse Data:")
print("-" * 50)

lasso_cv_sparse = LassoCV(alphas=alphas, cv=10, random_state=42, max_iter=2000)
lasso_cv_sparse.fit(X_scaled, Y_new)

lasso_final_sparse = Lasso(alpha=lasso_cv_sparse.alpha_, max_iter=2000)
lasso_final_sparse.fit(X_scaled, Y_new)

print(f"Optimal λ: {lasso_cv_sparse.alpha_:.6f}")

selected_lasso_sparse = []
print(f"Coefficients:")
for i, (coef, name) in enumerate(zip(lasso_final_sparse.coef_, feature_names)):
    print(f"{name}: {coef:.4f}")
    if abs(coef) > 1e-6:
        selected_lasso_sparse.append(name)

print(f"Selected features: {', '.join(selected_lasso_sparse)}")

# Analysis of sparse model results
print_section_header("SPARSE MODEL ANALYSIS AND DISCUSSION")

forward_sparse_features = set(sparse_best_forward['selected_names'])
lasso_sparse_features = set(selected_lasso_sparse)
true_sparse_feature = {'X^7'}

print(f"True model: Y = {beta0_new} + {beta7}*X^7 + ε")
print(f"True features: {', '.join(true_sparse_feature)}")
print(f"Forward stepwise selected: {', '.join(sorted(forward_sparse_features))}")
print(f"Lasso selected: {', '.join(sorted(lasso_sparse_features))}")

# Evaluation metrics
forward_correct = true_sparse_feature.issubset(forward_sparse_features)
lasso_correct = true_sparse_feature.issubset(lasso_sparse_features)

forward_false_pos = len(forward_sparse_features - true_sparse_feature)
lasso_false_pos = len(lasso_sparse_features - true_sparse_feature)

print(f"\nPerformance Analysis:")
print(f"Correct identification of X^7:")
print(f"  Forward stepwise: {'✓' if forward_correct else '✗'}")
print(f"  Lasso: {'✓' if lasso_correct else '✗'}")

print(f"\nFalse positives (irrelevant features selected):")
print(f"  Forward stepwise: {forward_false_pos}")
print(f"  Lasso: {lasso_false_pos}")

print(f"\nDiscussion:")
if lasso_false_pos < forward_false_pos:
    print("• Lasso performed better in the sparse setting, selecting fewer irrelevant features")
elif forward_false_pos < lasso_false_pos:
    print("• Forward stepwise performed better in the sparse setting")
else:
    print("• Both methods selected the same number of irrelevant features")

if forward_correct and lasso_correct:
    print("• Both methods successfully identified the true predictor X^7")
elif forward_correct:
    print("• Only forward stepwise correctly identified X^7")
elif lasso_correct:
    print("• Only lasso correctly identified X^7")
else:
    print("• Neither method correctly identified X^7")

print("\nKey Insights:")
print("• Lasso regularization helps with feature selection in high-dimensional settings")
print("• Stepwise selection can be sensitive to multicollinearity among polynomial terms")
print("• The sparse model scenario demonstrates the bias-variance tradeoff in model selection")

print("\n" + "=" * 70)
print("EXERCISE 8 COMPLETED SUCCESSFULLY!")
print("=" * 70)