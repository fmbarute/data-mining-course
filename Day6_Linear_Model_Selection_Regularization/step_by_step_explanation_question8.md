# Step-by-Step Explanation: Exercise 8 Implementation

## Section 1: Imports and Setup (Lines 1-23)

```python
#!/usr/bin/env python3
```
**Line 1**: Shebang line - tells the system to use python3 to execute this script

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```
**Lines 6-9**: Import essential libraries:
- `numpy`: For numerical operations and arrays
- `pandas`: For data manipulation (though not heavily used here)
- `matplotlib.pyplot`: For creating plots and visualizations
- `seaborn`: For enhanced statistical plotting

```python
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
```
**Lines 10-12**: Import scikit-learn components:
- `Lasso`: Lasso regression implementation
- `LassoCV`: Lasso with built-in cross-validation
- `StandardScaler`: For feature standardization
- `mean_squared_error`: For calculating MSE

```python
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table
```
**Lines 13-14**: Import statsmodels for detailed statistical analysis:
- `sm`: Main statsmodels API for regression
- `summary_table`: For detailed model diagnostics (imported but not used)

```python
warnings.filterwarnings('ignore')
np.random.seed(42)
```
**Lines 17-20**: 
- Suppress warning messages for cleaner output
- Set random seed for reproducible results

```python
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
```
**Lines 22-25**: Configure plotting aesthetics:
- Use default matplotlib style
- Set seaborn color palette to "husl" (evenly spaced hues)
- Set default figure size to 12x8 inches

---

## Section 2: StepwiseRegression Class Definition (Lines 27-67)

```python
class StepwiseRegression:
    """Implementation of forward and backward stepwise selection"""
    
    def __init__(self, method='cp'):
        self.method = method
        self.results = []
```
**Lines 27-32**: 
- Define the main class for stepwise selection
- `__init__` method initializes the class
- `method` parameter determines selection criterion ('cp', 'aic', or 'bic')
- `results` list will store step-by-step selection results

```python
def mallows_cp(self, y_true, y_pred, p, sigma2_full):
    """Calculate Mallows' Cp statistic"""
    n = len(y_true)
    mse = mean_squared_error(y_true, y_pred)
    cp = (mse / sigma2_full) * n - n + 2 * p
    return cp
```
**Lines 34-39**: 
- Method to calculate Mallows' Cp statistic
- `y_true`: actual response values
- `y_pred`: predicted values from model
- `p`: number of parameters (including intercept)
- `sigma2_full`: MSE from full model
- Formula: Cp = (MSE/σ²full) × n - n + 2p

```python
def aic(self, y_true, y_pred, p):
    """Calculate AIC"""
    n = len(y_true)
    mse = mean_squared_error(y_true, y_pred)
    return n * np.log(mse) + 2 * p
```
**Lines 41-45**: 
- Method to calculate Akaike Information Criterion (AIC)
- Formula: AIC = n × ln(MSE) + 2p
- Lower AIC indicates better model

```python
def bic(self, y_true, y_pred, p):
    """Calculate BIC"""
    n = len(y_true)
    mse = mean_squared_error(y_true, y_pred)
    return n * np.log(mse) + p * np.log(n)
```
**Lines 47-51**: 
- Method to calculate Bayesian Information Criterion (BIC)
- Formula: BIC = n × ln(MSE) + p × ln(n)
- More conservative than AIC (higher penalty for complexity)

---

## Section 3: Forward Selection Method (Lines 53-130)

```python
def forward_selection(self, X, y, feature_names=None):
    """Forward stepwise selection"""
    n, p = X.shape
    if feature_names is None:
        feature_names = [f'X{i+1}' for i in range(p)]
```
**Lines 53-57**: 
- Define forward selection method
- Get dimensions: n (observations), p (features)
- Create default feature names if not provided

```python
# Fit full model for Cp calculation
X_full = sm.add_constant(X)
full_model = sm.OLS(y, X_full).fit()
sigma2_full = full_model.mse_resid
```
**Lines 59-62**: 
- Add intercept column to feature matrix
- Fit full model using all features
- Extract MSE residual for Cp calculations

```python
# Initialize
selected = []
remaining = list(range(p))
results = []
```
**Lines 64-67**: 
- `selected`: list of selected feature indices
- `remaining`: list of remaining feature indices to consider
- `results`: list to store results from each step

```python
print("FORWARD STEPWISE SELECTION")
print("=" * 80)
print(f"{'Step':<4} {'Added':<10} {'Cp':<10} {'AIC':<10} {'BIC':<10} {'Selected Features'}")
print("-" * 80)
```
**Lines 69-72**: Print formatted header for output table

```python
for step in range(p):
    best_score = float('inf')
    best_feature = None
    best_stats = {}
```
**Lines 74-77**: 
- Loop through each potential step (max p steps)
- Initialize variables to track best feature to add
- `best_score`: best criterion value found so far
- `best_feature`: index of best feature to add
- `best_stats`: dictionary to store all criteria values

```python
# Try adding each remaining feature
for feature in remaining:
    trial_features = selected + [feature]
    X_trial = sm.add_constant(X[:, trial_features])
```
**Lines 79-82**: 
- Loop through each remaining feature
- Create trial feature set by adding current feature
- Prepare design matrix with intercept

```python
try:
    model = sm.OLS(y, X_trial).fit()
    y_pred = model.fittedvalues
    p_trial = len(trial_features) + 1  # +1 for intercept
```
**Lines 84-87**: 
- Fit OLS model with trial features
- Get predicted values
- Count parameters (features + intercept)

```python
# Calculate criteria
cp = self.mallows_cp(y, y_pred, p_trial, sigma2_full)
aic = self.aic(y, y_pred, p_trial)
bic = self.bic(y, y_pred, p_trial)
```
**Lines 89-92**: Calculate all three selection criteria

```python
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
```
**Lines 94-103**: 
- Choose score based on selected method
- Update best feature if current score is better
- Store all statistics for reporting

```python
except np.linalg.LinAlgError:
    continue
```
**Lines 105-106**: Handle linear algebra errors (e.g., singular matrices)

```python
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
```
**Lines 108-120**: 
- If a best feature was found, add it to selected list
- Remove from remaining features
- Create readable feature names
- Store detailed results for this step

```python
print(f"{step+1:<4} {feature_names[best_feature]:<10} "
      f"{best_stats['cp']:<10.2f} {best_stats['aic']:<10.2f} "
      f"{best_stats['bic']:<10.2f} {', '.join(selected_names)}")
```
**Lines 122-124**: Print formatted results for this step

```python
else:
    break
```
**Lines 125-126**: If no feature improves the model, stop

```python
# Find best model
best_model = min(results, key=lambda x: x[self.method])
print(f"\nBest model (Step {best_model['step']}, {self.method.upper()} = {best_model[self.method]:.2f}):")
print(f"Features: {', '.join(best_model['selected_names'])}")

return results, best_model
```
**Lines 128-132**: 
- Find the model with best (minimum) criterion value
- Print summary of best model
- Return all results and best model

---

## Section 4: Backward Selection Method (Lines 134-227)

The backward selection method follows similar logic but in reverse:

```python
def backward_selection(self, X, y, feature_names=None):
    # Start with all features
    selected = list(range(p))
```
**Lines 134-147**: 
- Initialize with ALL features selected
- Same setup as forward selection

```python
# Initial model (full model)
y_pred = full_model.fittedvalues
p_full = p + 1  # +1 for intercept
cp_init = self.mallows_cp(y, y_pred, p_full, sigma2_full)
```
**Lines 160-170**: 
- Start with full model
- Calculate initial criteria values
- Store initial results

```python
# Remove features one by one
for step in range(p):
    if len(selected) <= 1:
        break
```
**Lines 172-175**: 
- Loop to remove features
- Stop if only one feature remains

```python
# Try removing each feature
for feature in selected:
    trial_features = [f for f in selected if f != feature]
```
**Lines 181-183**: 
- Try removing each currently selected feature
- Create trial set without current feature

The rest follows the same pattern as forward selection but removes features instead of adding them.

---

## Section 5: Helper Functions (Lines 229-269)

```python
def print_section_header(title):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(f"{title}")
    print("=" * 70)
```
**Lines 229-233**: Simple function to print formatted section headers

```python
def fit_and_report_model(X, y, selected_indices, feature_names, title="Model"):
    """Fit final model and report detailed results"""
    if len(selected_indices) == 0:
        print(f"\n{title}: No features selected!")
        return None
```
**Lines 235-239**: 
- Function to fit final model and report detailed statistics
- Handle case where no features are selected

```python
# Fit the model
X_selected = X[:, selected_indices]
X_with_const = sm.add_constant(X_selected)
model = sm.OLS(y, X_with_const).fit()
```
**Lines 241-244**: 
- Extract selected features from feature matrix
- Add intercept column
- Fit OLS model

```python
print(f"\n{title} Results:")
print("-" * 50)
print(f"Selected features: {', '.join([feature_names[i] for i in selected_indices])}")
print(f"R-squared: {model.rsquared:.4f}")
print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
print(f"MSE: {model.mse_resid:.4f}")
print(f"AIC: {model.aic:.2f}")
print(f"BIC: {model.bic:.2f}")
```
**Lines 246-253**: Print comprehensive model statistics

```python
print(f"\nCoefficients:")
print(f"Intercept: {model.params[0]:.4f} (p-value: {model.pvalues[0]:.4f})")
for i, idx in enumerate(selected_indices):
    coef = model.params[i + 1]
    pval = model.pvalues[i + 1]
    print(f"{feature_names[idx]}: {coef:.4f} (p-value: {pval:.4f})")
```
**Lines 255-260**: 
- Print coefficient estimates
- Include p-values for statistical significance
- Loop through selected features

---

## Section 6: Data Generation - Parts A & B (Lines 271-293)

```python
# PART (A) and (B): Data Generation
print_section_header("PARTS (A) & (B): DATA GENERATION")

# Generate predictor X and noise ε
n = 100
X = np.random.normal(0, 1, n)
epsilon = np.random.normal(0, 1, n)
```
**Lines 271-277**: 
- Print section header
- Set sample size to 100
- Generate predictor X from standard normal distribution
- Generate noise ε from standard normal distribution

```python
# Generate response Y with chosen coefficients
beta0, beta1, beta2, beta3 = 3.0, 2.0, -1.5, 0.5
Y = beta0 + beta1 * X + beta2 * (X**2) + beta3 * (X**3) + epsilon
```
**Lines 279-281**: 
- Define true coefficients for the polynomial model
- Generate response Y according to: Y = 3.0 + 2.0X - 1.5X² + 0.5X³ + ε

```python
print(f"Sample size: n = {n}")
print(f"True model: Y = {beta0} + {beta1}*X + {beta2}*X² + {beta3}*X³ + ε")
print(f"X statistics: mean = {X.mean():.3f}, std = {X.std():.3f}")
print(f"Y statistics: mean = {Y.mean():.3f}, std = {Y.std():.3f}")
```
**Lines 283-286**: Print summary statistics of generated data

```python
# Create polynomial features X, X², ..., X^10
X_poly = np.column_stack([X**i for i in range(1, 11)])
feature_names = [f'X^{i}' for i in range(1, 11)]

print(f"Created polynomial features: {', '.join(feature_names)}")
```
**Lines 288-292**: 
- Create polynomial feature matrix (X¹, X², ..., X¹⁰)
- Use `np.column_stack` to combine feature columns
- Create descriptive feature names
- Print confirmation

---

## Section 7: Parts C & D - Stepwise Selection (Lines 294-331)

```python
# PART (C): Forward Stepwise Selection
print_section_header("PART (C): FORWARD STEPWISE SELECTION")

forward_selector = StepwiseRegression(method='cp')
forward_results, best_forward = forward_selector.forward_selection(X_poly, Y, feature_names)
```
**Lines 294-298**: 
- Execute Part C
- Create StepwiseRegression instance using Cp criterion
- Run forward selection and store results

```python
# Fit and report final forward model
forward_model = fit_and_report_model(X_poly, Y, best_forward['selected_indices'], 
                                   feature_names, "Forward Selection Final Model")
```
**Lines 300-302**: Fit and report the final forward selection model

```python
# PART (D): Backward Stepwise Selection
print_section_header("PART (D): BACKWARD STEPWISE SELECTION")

backward_selector = StepwiseRegression(method='cp')
backward_results, best_backward = backward_selector.backward_selection(X_poly, Y, feature_names)
```
**Lines 304-308**: Same process for backward selection

```python
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
```
**Lines 314-331**: 
- Compare results from forward and backward selection
- Convert to sets for easy set operations
- Find common features, unique features, and differences
- Use set operations (&, -, ==) for comparison

---

## Section 8: Part E - Lasso Regression (Lines 333-398)

```python
# PART (E): Lasso Regression
print_section_header("PART (E): LASSO REGRESSION")

# Standardize features for lasso
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)
```
**Lines 333-338**: 
- Start Part E
- Create StandardScaler instance
- Standardize features (mean=0, std=1) - required for lasso

```python
# Lasso with cross-validation
alphas = np.logspace(-4, 2, 100)
lasso_cv = LassoCV(alphas=alphas, cv=10, random_state=42, max_iter=2000)
lasso_cv.fit(X_scaled, Y)
```
**Lines 340-343**: 
- Create range of alpha values (10⁻⁴ to 10²)
- Use logarithmic spacing for 100 values
- LassoCV performs 10-fold cross-validation
- Fit to find optimal alpha

```python
print(f"Optimal λ (alpha): {lasso_cv.alpha_:.6f}")
print(f"Cross-validation R²: {lasso_cv.score(X_scaled, Y):.4f}")

# Fit final lasso model
lasso_final = Lasso(alpha=lasso_cv.alpha_, max_iter=2000)
lasso_final.fit(X_scaled, Y)
```
**Lines 345-350**: 
- Print optimal lambda value
- Print CV score
- Fit final lasso model with optimal alpha

```python
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
```
**Lines 352-362**: 
- Print lasso results
- Loop through coefficients and feature names simultaneously
- Identify non-zero coefficients (selected features)
- Use threshold of 1e-6 to account for numerical precision

---

## Section 9: Visualization (Lines 364-444)

```python
# Create visualization for lasso
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
```
**Lines 364-365**: Create 2x2 subplot grid for comprehensive visualization

### Plot 1: Cross-Validation Error (Lines 367-378)
```python
# Plot 1: Cross-validation error
ax1.semilogx(lasso_cv.alphas_, lasso_cv.mse_path_.mean(axis=1), 'b-', alpha=0.8)
ax1.fill_between(lasso_cv.alphas_, 
                 lasso_cv.mse_path_.mean(axis=1) - lasso_cv.mse_path_.std(axis=1),
                 lasso_cv.mse_path_.mean(axis=1) + lasso_cv.mse_path_.std(axis=1),
                 alpha=0.3)
ax1.axvline(lasso_cv.alpha_, color='red', linestyle='--', 
            label=f'Optimal λ = {lasso_cv.alpha_:.4f}')
```
**Lines 367-374**: 
- Plot CV error vs. alpha on log scale
- Add confidence bands using standard deviation
- Mark optimal alpha with vertical line

### Plot 2: Coefficient Paths (Lines 380-400)
```python
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
```
**Lines 380-395**: 
- Create new alpha grid for smoother coefficient paths
- Fit lasso for each alpha value
- Store coefficients for each fit
- Plot coefficient vs. alpha for each feature

### Plot 3: Feature Selection Comparison (Lines 402-425)
```python
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
```
**Lines 402-415**: 
- Create binary matrix showing which features each method selected
- Rows represent methods, columns represent features
- 1 indicates feature selected, 0 indicates not selected

```python
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
```
**Lines 417-427**: 
- Display matrix as heatmap
- Add checkmarks for selected features
- Configure axis labels and title

### Plot 4: Model Comparison (Lines 429-444)
```python
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
```
**Lines 429-444**: 
- Create bar chart comparing Cp scores
- Add value labels on top of bars
- Apply tight layout and display plots

---

## Section 10: Part F - Sparse Model Analysis (Lines 446-530)

```python
# PART (F): Sparse Model Analysis
print_section_header("PART (F): SPARSE MODEL ANALYSIS")

# Generate new response with only X^7 term
np.random.seed(123)  # Different seed for new experiment
epsilon_new = np.random.normal(0, 1, n)
beta0_new, beta7 = 2.0, 0.4
Y_new = beta0_new + beta7 * (X**7) + epsilon_new
```
**Lines 446-453**: 
- Start Part F analysis
- Use different random seed for new experiment
- Generate new noise vector
- Create sparse model with only X⁷ term: Y = 2.0 + 0.4X⁷ + ε

```python
print(f"New sparse model: Y = {beta0_new} + {beta7}*X^7 + ε")
print(f"Y_new statistics: mean = {Y_new.mean():.3f}, std = {Y_new.std():.3f}")

# Forward stepwise on sparse data
print(f"\nForward Stepwise Selection on Sparse Data:")
forward_sparse = StepwiseRegression(method='cp')
sparse_forward_results, sparse_best_forward = forward_sparse.forward_selection(X_poly, Y_new, feature_names)
```
**Lines 455-461**: 
- Print model specification and statistics
- Apply forward stepwise selection to sparse data

```python
# Lasso on sparse data
print(f"\nLasso Regression on Sparse Data:")
print("-" * 50)

lasso_cv_sparse = LassoCV(alphas=alphas, cv=10, random_state=42, max_iter=2000)
lasso_cv_sparse.fit(X_scaled, Y_new)

lasso_final_sparse = Lasso(alpha=lasso_cv_sparse.alpha_, max_iter=2000)
lasso_final_sparse.fit(X_scaled, Y_new)
```
**Lines 463-470**: 
- Apply lasso regression to sparse data
- Use same alpha grid and CV procedure
- Fit final model with optimal alpha

```python
print(f"Optimal λ: {lasso_cv_sparse.alpha_:.6f}")

selected_lasso_sparse = []
print(f"Coefficients:")
for i, (coef, name) in enumerate(zip(lasso_final_sparse.coef_, feature_names)):
    print(f"{name}: {coef:.4f}")
    if abs(coef) > 1e-6:
        selected_lasso_sparse.append(name)

print(f"Selected features: {', '.join(selected_lasso_sparse)}")
```
**Lines 472-480**: Report lasso results for sparse model

---

## Section 11: Final Analysis and Discussion (Lines 482-530)

```python
# Analysis of sparse model results
print_section_header("SPARSE MODEL ANALYSIS AND DISCUSSION")

forward_sparse_features = set(sparse_best_forward['selected_names'])
lasso_sparse_features = set(selected_lasso_sparse)
true_sparse_feature = {'X^7'}

print(f"True model: Y = {beta0_new} + {beta7}*X^7 + ε")
print(f"True features: {', '.join(true_sparse_feature)}")
print(f"Forward stepwise selected: {', '.join(sorted(forward_sparse_features))}")
print(f"Lasso selected: {', '.join(sorted(lasso_sparse_features))}")
```
**Lines 482-491**: 
- Convert selected features to sets for comparison
- Define true sparse feature set
- Print comparison of what each method selected

```python
# Evaluation metrics
forward_correct = true_sparse_feature.issubset(forward_sparse_features)
lasso_correct = true_sparse_feature.issubset(lasso_sparse_features)

forward_false_pos = len(forward_sparse_features - true_sparse_feature)
lasso_false_pos = len(lasso_sparse_features - true_sparse_feature)
```
**Lines 493-497**: 
- Check if each method correctly identified X⁷
- Count false positives (irrelevant features selected)
- Use set operations for clean comparisons

```python
print(f"\nPerformance Analysis:")
print(f"Correct identification of X^7:")
print(f"  Forward stepwise: {'✓' if forward_correct else '✗'}")
print(f"  Lasso: {'✓' if lasso_correct else '✗'}")

print(f"\nFalse positives (irrelevant features selected):")
print(f"  Forward stepwise: {forward_false_pos}")
print(f"  Lasso: {lasso_false_pos}")
```
**Lines 499-506**: Print performance metrics with clear symbols

```python
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
```
**Lines 508-521**: 
- Provide automated analysis based on results
- Compare method performance
- Give context about correct identification

```python
print("\nKey Insights:")
print("• Lasso regularization helps with feature selection in high-dimensional settings")
print("• Stepwise selection can be sensitive to multicollinearity among polynomial terms")
print("• The sparse model scenario demonstrates the bias-variance tradeoff in model selection")

print("\n" + "=" * 70)
print("EXERCISE 8 COMPLETED SUCCESSFULLY!")
print("=" * 70)
```
**Lines 523-530**: 
- Provide general insights about the methods
- Print completion message

---

## Summary

This script provides a complete implementation of Exercise 8 with:

1. **Custom stepwise selection algorithms** with proper statistical criteria
2. **Comprehensive comparison** between forward, backward, and lasso methods
3. **Detailed visualization** showing selection paths and performance
4. **Two scenarios**: dense model (X, X², X³) and sparse model (X⁷ only)
5. **Automated analysis** comparing method effectiveness

The code demonstrates the practical differences between these model selection approaches and their performance in different scenarios.