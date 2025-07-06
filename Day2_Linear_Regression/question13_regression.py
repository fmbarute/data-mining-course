# Exercise 13: Simulated Linear Regression Analysis
# Chapter 3: Linear Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

print("📊 Exercise 13: Simulated Linear Regression Analysis")
print("=" * 60)

# Set random seed for reproducibility (as required)
np.random.seed(1)
print("🎲 Random seed set to 1 for consistent results")

# ============================================================================
# Part (a): Create feature vector X
# ============================================================================
print("\n" + "=" * 60)
print("(a) CREATING FEATURE VECTOR X")
print("=" * 60)

# Create vector x with 100 observations from N(0,1)
n = 100
x = np.random.normal(0, 1, n)

print(f"✅ Created vector x with {len(x)} observations")
print(f"📊 x ~ N(0, 1)")
print(f"📈 x statistics:")
print(f"   • Mean: {np.mean(x):.4f}")
print(f"   • Standard deviation: {np.std(x):.4f}")
print(f"   • Min: {np.min(x):.4f}")
print(f"   • Max: {np.max(x):.4f}")

# ============================================================================
# Part (b): Create error term ε
# ============================================================================
print("\n" + "=" * 60)
print("(b) CREATING ERROR TERM ε")
print("=" * 60)

# Create error vector eps with 100 observations from N(0, 0.25)
# Note: variance = 0.25, so standard deviation = sqrt(0.25) = 0.5
eps = np.random.normal(0, np.sqrt(0.25), n)

print(f"✅ Created error vector eps with {len(eps)} observations")
print(f"📊 eps ~ N(0, 0.25)")
print(f"📈 eps statistics:")
print(f"   • Mean: {np.mean(eps):.4f}")
print(f"   • Variance: {np.var(eps):.4f}")
print(f"   • Standard deviation: {np.std(eps):.4f}")
print(f"   • Theoretical std: {np.sqrt(0.25):.4f}")

# ============================================================================
# Part (c): Generate response variable Y
# ============================================================================
print("\n" + "=" * 60)
print("(c) GENERATING RESPONSE VARIABLE Y")
print("=" * 60)

# Generate y according to Y = -1 + 0.5X + ε
beta_0 = -1  # True intercept
beta_1 = 0.5  # True slope
y = beta_0 + beta_1 * x + eps

print(f"✅ Generated y according to model: Y = -1 + 0.5X + ε")
print(f"📏 Length of vector y: {len(y)}")
print(f"📊 True parameters:")
print(f"   • β₀ (intercept): {beta_0}")
print(f"   • β₁ (slope): {beta_1}")
print(f"📈 y statistics:")
print(f"   • Mean: {np.mean(y):.4f}")
print(f"   • Standard deviation: {np.std(y):.4f}")
print(f"   • Min: {np.min(y):.4f}")
print(f"   • Max: {np.max(y):.4f}")

# ============================================================================
# Part (d): Create scatterplot
# ============================================================================
print("\n" + "=" * 60)
print("(d) SCATTERPLOT OF X vs Y")
print("=" * 60)

plt.figure(figsize=(10, 8))
plt.scatter(x, y, alpha=0.7, color='steelblue', s=50)
plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.title('Scatterplot: X vs Y\nTrue model: Y = -1 + 0.5X + ε', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Add true population line for reference
x_line = np.linspace(np.min(x), np.max(x), 100)
y_true = beta_0 + beta_1 * x_line
plt.plot(x_line, y_true, color='red', linewidth=2, linestyle='--',
         label=f'True line: Y = {beta_0} + {beta_1}X', alpha=0.8)
plt.legend()
plt.tight_layout()
plt.show()

print(f"✅ Scatterplot created!")
print(f"🔍 OBSERVATIONS:")
print(f"   • Clear linear relationship between X and Y")
print(f"   • Points scattered around the true regression line")
print(f"   • Positive slope as expected (β₁ = 0.5)")
print(f"   • Random scatter due to error term ε")
print(f"   • Linear relationship appears strong despite noise")

# ============================================================================
# Part (e): Fit least squares linear model
# ============================================================================
print("\n" + "=" * 60)
print("(e) LEAST SQUARES LINEAR MODEL")
print("=" * 60)

# Fit least squares model using statsmodels
X_with_const = sm.add_constant(x)
model = sm.OLS(y, X_with_const)
results = model.fit()

# Extract fitted coefficients
beta_0_hat = results.params['const']
beta_1_hat = results.params['x1']

print(f"📊 REGRESSION RESULTS:")
print(results.summary())

print(f"\n🔍 COEFFICIENT COMPARISON:")
print(f"{'Parameter':<12} {'True Value':<12} {'Estimated':<12} {'Difference':<12}")
print(f"{'-' * 50}")
print(f"{'β₀ (intercept)':<12} {beta_0:<12.4f} {beta_0_hat:<12.4f} {abs(beta_0 - beta_0_hat):<12.4f}")
print(f"{'β₁ (slope)':<12} {beta_1:<12.4f} {beta_1_hat:<12.4f} {abs(beta_1 - beta_1_hat):<12.4f}")

print(f"\n📈 MODEL ASSESSMENT:")
print(f"   • R-squared: {results.rsquared:.4f}")
print(f"   • Adjusted R-squared: {results.rsquared_adj:.4f}")
print(f"   • F-statistic: {results.fvalue:.2f}")
print(f"   • p-value: {results.f_pvalue:.2e}")

# Assessment
if abs(beta_0 - beta_0_hat) < 0.1 and abs(beta_1 - beta_1_hat) < 0.1:
    print(f"✅ Excellent fit: Estimated coefficients very close to true values")
elif abs(beta_0 - beta_0_hat) < 0.2 and abs(beta_1 - beta_1_hat) < 0.2:
    print(f"✅ Good fit: Estimated coefficients reasonably close to true values")
else:
    print(f"⚠️ Moderate fit: Some difference between estimated and true coefficients")

# ============================================================================
# Part (f): Display least squares line with population regression line
# ============================================================================
print("\n" + "=" * 60)
print("(f) REGRESSION LINES COMPARISON")
print("=" * 60)

plt.figure(figsize=(12, 8))

# Scatter plot
plt.scatter(x, y, alpha=0.7, color='steelblue', s=50, label='Data points')

# True population regression line
x_line = np.linspace(np.min(x), np.max(x), 100)
y_true = beta_0 + beta_1 * x_line
plt.plot(x_line, y_true, color='red', linewidth=3, linestyle='--',
         label=f'True line: Y = {beta_0} + {beta_1}X', alpha=0.9)

# Fitted least squares line
y_fitted = beta_0_hat + beta_1_hat * x_line
plt.plot(x_line, y_fitted, color='green', linewidth=3,
         label=f'Fitted line: Y = {beta_0_hat:.3f} + {beta_1_hat:.3f}X', alpha=0.9)

plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.title('Comparison: True vs Fitted Regression Lines', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"✅ Regression lines comparison plot created!")
print(f"🔍 VISUAL ASSESSMENT:")
print(f"   • Green line (fitted) should be very close to red line (true)")
print(f"   • Both lines should pass through the center of the data cloud")
print(f"   • Small differences due to random sampling variation")

# ============================================================================
# Part (g): Polynomial regression model
# ============================================================================
print("\n" + "=" * 60)
print("(g) POLYNOMIAL REGRESSION MODEL")
print("=" * 60)

# Fit polynomial model with x and x²
X_poly = np.column_stack([x, x ** 2])
X_poly_with_const = sm.add_constant(X_poly)
model_poly = sm.OLS(y, X_poly_with_const)
results_poly = model_poly.fit()

print(f"📊 POLYNOMIAL REGRESSION RESULTS:")
print(results_poly.summary())

# Compare models
print(f"\n🔍 MODEL COMPARISON:")
print(f"{'Metric':<20} {'Linear Model':<15} {'Polynomial Model':<18} {'Improvement':<12}")
print(f"{'-' * 70}")
print(
    f"{'R-squared':<20} {results.rsquared:<15.4f} {results_poly.rsquared:<18.4f} {results_poly.rsquared - results.rsquared:<12.4f}")
print(
    f"{'Adj R-squared':<20} {results.rsquared_adj:<15.4f} {results_poly.rsquared_adj:<18.4f} {results_poly.rsquared_adj - results.rsquared_adj:<12.4f}")
print(f"{'AIC':<20} {results.aic:<15.1f} {results_poly.aic:<18.1f} {results_poly.aic - results.aic:<12.1f}")
print(f"{'BIC':<20} {results.bic:<15.1f} {results_poly.bic:<18.1f} {results_poly.bic - results.bic:<12.1f}")

# Test significance of quadratic term
quad_coef = results_poly.params.iloc[2]  # x² coefficient
quad_pvalue = results_poly.pvalues.iloc[2]

print(f"\n📈 QUADRATIC TERM ANALYSIS:")
print(f"   • x² coefficient: {quad_coef:.6f}")
print(f"   • p-value: {quad_pvalue:.4f}")
print(f"   • Standard error: {results_poly.bse.iloc[2]:.6f}")

if quad_pvalue < 0.05:
    print(f"✅ Quadratic term is statistically significant (p < 0.05)")
    print(f"   Evidence that quadratic term improves model fit")
else:
    print(f"❌ Quadratic term is NOT statistically significant (p ≥ 0.05)")
    print(f"   No evidence that quadratic term improves model fit")

print(f"\n💡 INTERPRETATION:")
print(f"   • The true model is linear: Y = -1 + 0.5X + ε")
print(f"   • Adding x² should not significantly improve fit")
print(f"   • Any improvement is likely due to overfitting")

# ============================================================================
# Part (h): Less noise scenario
# ============================================================================
print("\n" + "=" * 60)
print("(h) LESS NOISE SCENARIO")
print("=" * 60)

# Reset seed for consistency
np.random.seed(1)
x_low = np.random.normal(0, 1, n)

# Lower variance error term (variance = 0.05 instead of 0.25)
eps_low = np.random.normal(0, np.sqrt(0.05), n)
y_low = beta_0 + beta_1 * x_low + eps_low

# Fit model
X_low_const = sm.add_constant(x_low)
model_low = sm.OLS(y_low, X_low_const)
results_low = model_low.fit()

beta_0_hat_low = results_low.params['const']
beta_1_hat_low = results_low.params['x1']

print(f"📊 LESS NOISE MODEL RESULTS:")
print(f"   • Error variance: 0.05 (vs original 0.25)")
print(f"   • Error std dev: {np.sqrt(0.05):.3f} (vs original {np.sqrt(0.25):.3f})")

print(f"\n🔍 COEFFICIENT COMPARISON (Less Noise):")
print(f"{'Parameter':<12} {'True Value':<12} {'Estimated':<12} {'Difference':<12}")
print(f"{'-' * 50}")
print(f"{'β₀ (intercept)':<12} {beta_0:<12.4f} {beta_0_hat_low:<12.4f} {abs(beta_0 - beta_0_hat_low):<12.4f}")
print(f"{'β₁ (slope)':<12} {beta_1:<12.4f} {beta_1_hat_low:<12.4f} {abs(beta_1 - beta_1_hat_low):<12.4f}")

print(f"\n📈 MODEL PERFORMANCE (Less Noise):")
print(f"   • R-squared: {results_low.rsquared:.4f}")
print(f"   • Standard errors smaller due to less noise")
print(f"   • Estimates should be more precise")

# ============================================================================
# Part (i): More noise scenario
# ============================================================================
print("\n" + "=" * 60)
print("(i) MORE NOISE SCENARIO")
print("=" * 60)

# Reset seed for consistency
np.random.seed(1)
x_high = np.random.normal(0, 1, n)

# Higher variance error term (variance = 1.0 instead of 0.25)
eps_high = np.random.normal(0, np.sqrt(1.0), n)
y_high = beta_0 + beta_1 * x_high + eps_high

# Fit model
X_high_const = sm.add_constant(x_high)
model_high = sm.OLS(y_high, X_high_const)
results_high = model_high.fit()

beta_0_hat_high = results_high.params['const']
beta_1_hat_high = results_high.params['x1']

print(f"📊 MORE NOISE MODEL RESULTS:")
print(f"   • Error variance: 1.0 (vs original 0.25)")
print(f"   • Error std dev: {np.sqrt(1.0):.3f} (vs original {np.sqrt(0.25):.3f})")

print(f"\n🔍 COEFFICIENT COMPARISON (More Noise):")
print(f"{'Parameter':<12} {'True Value':<12} {'Estimated':<12} {'Difference':<12}")
print(f"{'-' * 50}")
print(f"{'β₀ (intercept)':<12} {beta_0:<12.4f} {beta_0_hat_high:<12.4f} {abs(beta_0 - beta_0_hat_high):<12.4f}")
print(f"{'β₁ (slope)':<12} {beta_1:<12.4f} {beta_1_hat_high:<12.4f} {abs(beta_1 - beta_1_hat_high):<12.4f}")

print(f"\n📈 MODEL PERFORMANCE (More Noise):")
print(f"   • R-squared: {results_high.rsquared:.4f}")
print(f"   • Standard errors larger due to more noise")
print(f"   • Estimates should be less precise")

# ============================================================================
# Comparison visualization for parts (h) and (i)
# ============================================================================
print("\n📊 Creating comparison visualization...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

scenarios = [
    (x, y, results, "Original (σ² = 0.25)", 'steelblue'),
    (x_low, y_low, results_low, "Less Noise (σ² = 0.05)", 'green'),
    (x_high, y_high, results_high, "More Noise (σ² = 1.0)", 'orange')
]

for i, (x_data, y_data, model_results, title, color) in enumerate(scenarios):
    axes[i].scatter(x_data, y_data, alpha=0.6, color=color, s=40)

    # True line
    x_line = np.linspace(np.min(x_data), np.max(x_data), 100)
    y_true = beta_0 + beta_1 * x_line
    axes[i].plot(x_line, y_true, 'r--', linewidth=2, label='True line', alpha=0.8)

    # Fitted line
    beta_0_fit = model_results.params['const']
    beta_1_fit = model_results.params['x1']
    y_fitted = beta_0_fit + beta_1_fit * x_line
    axes[i].plot(x_line, y_fitted, 'black', linewidth=2, label='Fitted line')

    axes[i].set_xlabel('X')
    axes[i].set_ylabel('Y')
    axes[i].set_title(f'{title}\nR² = {model_results.rsquared:.3f}')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# Part (j): Confidence intervals comparison
# ============================================================================
print("\n" + "=" * 60)
print("(j) CONFIDENCE INTERVALS COMPARISON")
print("=" * 60)

# Extract confidence intervals for all three scenarios
ci_original = results.conf_int()
ci_low = results_low.conf_int()
ci_high = results_high.conf_int()

print(f"📊 95% CONFIDENCE INTERVALS:")
print(f"\n{'Scenario':<15} {'Parameter':<12} {'Lower Bound':<12} {'Upper Bound':<12} {'Width':<12}")
print(f"{'-' * 65}")

# Original scenario
ci_width_orig_b0 = ci_original.iloc[0, 1] - ci_original.iloc[0, 0]
ci_width_orig_b1 = ci_original.iloc[1, 1] - ci_original.iloc[1, 0]

print(
    f"{'Original':<15} {'β₀':<12} {ci_original.iloc[0, 0]:<12.4f} {ci_original.iloc[0, 1]:<12.4f} {ci_width_orig_b0:<12.4f}")
print(
    f"{'(σ² = 0.25)':<15} {'β₁':<12} {ci_original.iloc[1, 0]:<12.4f} {ci_original.iloc[1, 1]:<12.4f} {ci_width_orig_b1:<12.4f}")

# Less noise scenario
ci_width_low_b0 = ci_low.iloc[0, 1] - ci_low.iloc[0, 0]
ci_width_low_b1 = ci_low.iloc[1, 1] - ci_low.iloc[1, 0]

print(f"{'Less Noise':<15} {'β₀':<12} {ci_low.iloc[0, 0]:<12.4f} {ci_low.iloc[0, 1]:<12.4f} {ci_width_low_b0:<12.4f}")
print(f"{'(σ² = 0.05)':<15} {'β₁':<12} {ci_low.iloc[1, 0]:<12.4f} {ci_low.iloc[1, 1]:<12.4f} {ci_width_low_b1:<12.4f}")

# More noise scenario
ci_width_high_b0 = ci_high.iloc[0, 1] - ci_high.iloc[0, 0]
ci_width_high_b1 = ci_high.iloc[1, 1] - ci_high.iloc[1, 0]

print(
    f"{'More Noise':<15} {'β₀':<12} {ci_high.iloc[0, 0]:<12.4f} {ci_high.iloc[0, 1]:<12.4f} {ci_width_high_b0:<12.4f}")
print(
    f"{'(σ² = 1.0)':<15} {'β₁':<12} {ci_high.iloc[1, 0]:<12.4f} {ci_high.iloc[1, 1]:<12.4f} {ci_width_high_b1:<12.4f}")

# Summary of findings
print(f"\n💡 CONFIDENCE INTERVAL ANALYSIS:")
print(f"✅ KEY FINDINGS:")
print(f"   • Less noise → Narrower confidence intervals → More precise estimates")
print(f"   • More noise → Wider confidence intervals → Less precise estimates")
print(f"   • All intervals should contain the true values (β₀ = -1, β₁ = 0.5)")


# Check if true values are in confidence intervals
def check_ci_coverage(ci, true_beta_0, true_beta_1):
    b0_covered = ci.iloc[0, 0] <= true_beta_0 <= ci.iloc[0, 1]
    b1_covered = ci.iloc[1, 0] <= true_beta_1 <= ci.iloc[1, 1]
    return b0_covered, b1_covered


b0_orig, b1_orig = check_ci_coverage(ci_original, beta_0, beta_1)
b0_low, b1_low = check_ci_coverage(ci_low, beta_0, beta_1)
b0_high, b1_high = check_ci_coverage(ci_high, beta_0, beta_1)

print(f"\n🎯 COVERAGE CHECK (True values in CIs?):")
print(f"   • Original: β₀ {'✅' if b0_orig else '❌'}, β₁ {'✅' if b1_orig else '❌'}")
print(f"   • Less noise: β₀ {'✅' if b0_low else '❌'}, β₁ {'✅' if b1_low else '❌'}")
print(f"   • More noise: β₀ {'✅' if b0_high else '❌'}, β₁ {'✅' if b1_high else '❌'}")

# ============================================================================
# Final Summary
# ============================================================================
print(f"\n" + "=" * 60)
print("📋 FINAL SUMMARY")
print("=" * 60)

print(f"✅ EXERCISE COMPLETION:")
print(f"   (a) ✅ Created feature vector X ~ N(0,1)")
print(f"   (b) ✅ Created error term ε ~ N(0,0.25)")
print(f"   (c) ✅ Generated Y = -1 + 0.5X + ε")
print(f"   (d) ✅ Created scatterplot showing linear relationship")
print(f"   (e) ✅ Fitted least squares model with good coefficient recovery")
print(f"   (f) ✅ Compared true vs fitted regression lines")
print(f"   (g) ✅ Tested polynomial model - no significant improvement")
print(f"   (h) ✅ Analyzed less noise scenario - more precise estimates")
print(f"   (i) ✅ Analyzed more noise scenario - less precise estimates")
print(f"   (j) ✅ Compared confidence intervals across noise levels")

print(f"\n🔍 KEY INSIGHTS:")
print(f"   • Least squares provides unbiased estimates of true parameters")
print(f"   • Noise level directly affects estimation precision")
print(f"   • Quadratic terms don't improve linear relationships")
print(f"   • Confidence intervals reflect estimation uncertainty")
print(f"   • Lower noise → Better model fit and narrower CIs")

print(f"\n📊 FINAL MODEL PERFORMANCE COMPARISON:")
print(f"{'Scenario':<15} {'R²':<8} {'β₀ Error':<10} {'β₁ Error':<10} {'CI Width (β₁)':<15}")
print(f"{'-' * 60}")
print(
    f"{'Original':<15} {results.rsquared:<8.4f} {abs(beta_0 - beta_0_hat):<10.4f} {abs(beta_1 - beta_1_hat):<10.4f} {ci_width_orig_b1:<15.4f}")
print(
    f"{'Less Noise':<15} {results_low.rsquared:<8.4f} {abs(beta_0 - beta_0_hat_low):<10.4f} {abs(beta_1 - beta_1_hat_low):<10.4f} {ci_width_low_b1:<15.4f}")
print(
    f"{'More Noise':<15} {results_high.rsquared:<8.4f} {abs(beta_0 - beta_0_hat_high):<10.4f} {abs(beta_1 - beta_1_hat_high):<10.4f} {ci_width_high_b1:<15.4f}")

print(f"\n🎉 Exercise 13 completed successfully!")
print(f"📚 This exercise demonstrates fundamental concepts in linear regression:")
print(f"   • Parameter estimation and recovery")
print(f"   • Effect of noise on model performance")
print(f"   • Model selection and overfitting")
print(f"   • Statistical inference and confidence intervals")