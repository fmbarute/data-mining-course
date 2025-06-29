
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
import warnings

warnings.filterwarnings('ignore')

print("🚗 Auto Dataset Simple Linear Regression Analysis")
print("=" * 60)


print("\n📁 Loading Auto dataset...")

try:
    # Try to load from OpenML (Auto dataset)
    from sklearn.datasets import fetch_openml

    auto = fetch_openml('autompg', version=1, as_frame=True, parser='auto')
    df = auto.frame

    # Clean and prepare data
    df = df.rename(columns={
        'mpg': 'mpg',
        'horsepower': 'horsepower',
        'weight': 'weight',
        'acceleration': 'acceleration',
        'model_year': 'year',
        'origin': 'origin'
    })

    # Handle missing values and convert horsepower to numeric
    df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
    df = df.dropna(subset=['mpg', 'horsepower'])
    print("✅ Real Auto dataset loaded!")

except:
    # Create synthetic Auto dataset if loading fails - FIXED VERSION
    print("Creating synthetic Auto dataset...")
    np.random.seed(42)
    n = 392

    # FIXED: Generate realistic horsepower values with proper variation
    horsepower = np.random.uniform(50, 250, n)  # Use uniform distribution for variation

    # FIXED: Generate mpg with realistic negative relationship to horsepower
    mpg = 40 - 0.15 * horsepower + np.random.normal(0, 3, n)
    mpg = np.clip(mpg, 5, 50)

    df = pd.DataFrame({
        'mpg': mpg,
        'horsepower': horsepower
    })
    print("✅ Synthetic dataset created!")

print(f"📊 Shape: {df.shape}")
print(f"📋 Variables: {list(df.columns)}")
print("\nFirst 5 rows:")
print(df[['mpg', 'horsepower']].head())

# Verify data has variation
print(f"\n🔍 Data Verification:")
print(f"• Horsepower range: {df['horsepower'].min():.1f} to {df['horsepower'].max():.1f}")
print(f"• MPG range: {df['mpg'].min():.1f} to {df['mpg'].max():.1f}")
print(f"• Horsepower std: {df['horsepower'].std():.2f}")

# ============================================================================
# Part (a): Simple Linear Regression with statsmodels - FIXED
# ============================================================================
print("\n" + "=" * 60)
print("(a) SIMPLE LINEAR REGRESSION: mpg ~ horsepower")
print("=" * 60)

# Prepare data for regression
X = df['horsepower']
y = df['mpg']

# Add constant term for intercept - FIXED
X_with_const = sm.add_constant(X)
print(f"✅ X matrix shape: {X_with_const.shape}")
print(f"✅ X matrix columns: {list(X_with_const.columns)}")

# Fit the model using OLS
model = sm.OLS(y, X_with_const)
results = model.fit()

# Print comprehensive results
print("\n📊 REGRESSION RESULTS:")
print(results.summary())

# FIXED: Extract key statistics with error handling
try:
    r_squared = results.rsquared
    adj_r_squared = results.rsquared_adj
    f_statistic = results.fvalue
    f_pvalue = results.f_pvalue

    # Check if const exists in parameters
    if 'const' in results.params.index:
        coef_intercept = results.params['const']
        se_intercept = results.bse['const']
    else:
        coef_intercept = 0
        se_intercept = 0
        print("⚠️ Warning: No intercept found in model")

    coef_horsepower = results.params['horsepower']
    se_horsepower = results.bse['horsepower']
    t_stat_horsepower = results.tvalues['horsepower']
    p_value_horsepower = results.pvalues['horsepower']

    print(f"\n🔍 KEY STATISTICAL INSIGHTS:")
    print(f"=" * 40)

    # (i) Is there a relationship?
    print(f"\n(i) RELATIONSHIP EXISTENCE:")
    print(f"• F-statistic: {f_statistic:.2f}")
    print(f"• F p-value: {f_pvalue:.2e}")
    print(f"• t-statistic (horsepower): {t_stat_horsepower:.2f}")
    print(f"• p-value (horsepower): {p_value_horsepower:.2e}")

    if p_value_horsepower < 0.05:
        print(f"✅ YES - Strong evidence of relationship (p < 0.05)")
    else:
        print(f"❌ NO - No significant relationship (p ≥ 0.05)")

    # (ii) How strong is the relationship?
    print(f"\n(ii) RELATIONSHIP STRENGTH:")
    print(f"• R-squared: {r_squared:.4f}")
    print(f"• Adjusted R-squared: {adj_r_squared:.4f}")
    print(f"• Correlation coefficient: {np.sqrt(r_squared) * (-1 if coef_horsepower < 0 else 1):.4f}")

    if r_squared > 0.7:
        strength = "Very Strong"
    elif r_squared > 0.5:
        strength = "Strong"
    elif r_squared > 0.3:
        strength = "Moderate"
    else:
        strength = "Weak"

    print(f"📈 Interpretation: {strength} relationship")
    print(f"   ({r_squared * 100:.1f}% of mpg variance explained by horsepower)")

    # (iii) Positive or negative relationship?
    print(f"\n(iii) RELATIONSHIP DIRECTION:")
    print(f"• Slope coefficient: {coef_horsepower:.6f}")
    print(f"• Standard error: {se_horsepower:.6f}")

    if coef_horsepower > 0:
        direction = "POSITIVE"
        interpretation = "mpg increases as horsepower increases"
    else:
        direction = "NEGATIVE"
        interpretation = "mpg decreases as horsepower increases"

    print(f"📊 Direction: {direction}")
    print(f"   Interpretation: {interpretation}")
    print(f"   For every 1 unit increase in horsepower, mpg changes by {coef_horsepower:.4f}")

    # (iv) Prediction with confidence and prediction intervals
    print(f"\n(iv) PREDICTION FOR HORSEPOWER = 98:")
    print(f"=" * 40)

    # Point prediction
    horsepower_new = 98
    X_new = sm.add_constant([horsepower_new])
    prediction = results.predict(X_new)[0]

    print(f"🎯 Point Prediction: {prediction:.2f} mpg")

    # 95% Confidence Interval for the mean response
    conf_int = results.get_prediction(X_new).conf_int(alpha=0.05)
    conf_lower, conf_upper = conf_int[0]

    print(f"📊 95% Confidence Interval: [{conf_lower:.2f}, {conf_upper:.2f}] mpg")
    print(f"   (Confidence interval for the MEAN mpg of all cars with 98 horsepower)")

    # 95% Prediction Interval for individual response
    pred_int = results.get_prediction(X_new).conf_int(alpha=0.05, obs=True)
    pred_lower, pred_upper = pred_int[0]

    print(f"🔮 95% Prediction Interval: [{pred_lower:.2f}, {pred_upper:.2f}] mpg")
    print(f"   (Prediction interval for a SINGLE car with 98 horsepower)")

    print(f"\n💡 Interval Interpretation:")
    print(f"• Confidence interval is NARROWER (uncertainty about the mean)")
    print(f"• Prediction interval is WIDER (includes individual car variability)")

except Exception as e:
    print(f"❌ Error in analysis: {e}")
    print("This usually means the data has no variation or model fitting failed")


print(f"\n" + "=" * 60)
print("(b) SCATTER PLOT WITH REGRESSION LINE")
print("=" * 60)

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Scatter plot
ax.scatter(df['horsepower'], df['mpg'], alpha=0.6, color='steelblue', s=50)

# Add regression line
x_range = np.linspace(df['horsepower'].min(), df['horsepower'].max(), 100)
y_pred = coef_intercept + coef_horsepower * x_range

ax.plot(x_range, y_pred, color='red', linewidth=2, label='Regression Line')

# Add prediction point for HP=98
if 'prediction' in locals():
    ax.scatter(horsepower_new, prediction, color='orange', s=100,
               label=f'Prediction (HP=98): {prediction:.1f} mpg', zorder=5)

# Formatting
ax.set_xlabel('Horsepower', fontsize=12)
ax.set_ylabel('Miles per Gallon (mpg)', fontsize=12)
ax.set_title('Simple Linear Regression: MPG vs Horsepower', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Add regression equation to plot
equation = f'mpg = {coef_intercept:.2f} + ({coef_horsepower:.4f}) × horsepower'
ax.text(0.05, 0.95, equation, transform=ax.transAxes, fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

# Add R-squared to plot
r_sq_text = f'R² = {r_squared:.3f}'
ax.text(0.05, 0.88, r_sq_text, transform=ax.transAxes, fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))

plt.tight_layout()
plt.show()

print(f"✅ Scatter plot with regression line created!")


print(f"\n" + "=" * 60)
print("(c) DIAGNOSTIC PLOTS")
print("=" * 60)


fitted_values = results.fittedvalues
residuals = results.resid
standardized_residuals = residuals / np.sqrt(results.mse_resid)

# Create diagnostic plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Residuals vs Fitted Values
axes[0, 0].scatter(fitted_values, residuals, alpha=0.6, color='steelblue')
axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.8)
axes[0, 0].set_xlabel('Fitted Values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Residuals vs Fitted Values')
axes[0, 0].grid(True, alpha=0.3)

# 2. Q-Q Plot (Normal probability plot)
stats.probplot(standardized_residuals, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Q-Q Plot (Normal Probability Plot)')
axes[0, 1].grid(True, alpha=0.3)

# 3. Scale-Location Plot
sqrt_abs_resid = np.sqrt(np.abs(standardized_residuals))
axes[1, 0].scatter(fitted_values, sqrt_abs_resid, alpha=0.6, color='steelblue')
axes[1, 0].set_xlabel('Fitted Values')
axes[1, 0].set_ylabel('√|Standardized Residuals|')
axes[1, 0].set_title('Scale-Location Plot')
axes[1, 0].grid(True, alpha=0.3)

# 4. Residuals vs Leverage
leverage = results.get_influence().hat_matrix_diag
axes[1, 1].scatter(leverage, standardized_residuals, alpha=0.6, color='steelblue')
axes[1, 1].set_xlabel('Leverage')
axes[1, 1].set_ylabel('Standardized Residuals')
axes[1, 1].set_title('Residuals vs Leverage')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Diagnostic Analysis
print(f"\n🔍 DIAGNOSTIC PLOT ANALYSIS:")
print(f"=" * 40)

# Basic diagnostic checks
try:
    # 1. Linearity
    residual_pattern = np.corrcoef(fitted_values, residuals)[0, 1]
    print(f"\n1️⃣ LINEARITY:")
    print(f"• Residuals vs Fitted correlation: {residual_pattern:.4f}")
    print(f"• Status: {'✅ Good' if abs(residual_pattern) < 0.1 else '⚠️ Check needed'}")

    # 2. Homoscedasticity
    fitted_median = np.median(fitted_values)
    var_low = np.var(residuals[fitted_values <= fitted_median])
    var_high = np.var(residuals[fitted_values > fitted_median])
    variance_ratio = var_high / var_low
    print(f"\n2️⃣ EQUAL VARIANCE:")
    print(f"• Variance ratio (high/low): {variance_ratio:.2f}")
    print(f"• Status: {'✅ Good' if 0.5 <= variance_ratio <= 2.0 else '⚠️ Check needed'}")

    # 3. Normality
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    print(f"\n3️⃣ NORMALITY:")
    print(f"• Shapiro-Wilk p-value: {shapiro_p:.4f}")
    print(f"• Status: {'✅ Good' if shapiro_p > 0.05 else '⚠️ Check needed'}")

except Exception as e:
    print(f"Diagnostic analysis error: {e}")

print(f"\n✅ SUMMARY:")
print(f"📈 R² = {r_squared:.3f} - {strength} relationship")
print(f"📊 Slope = {coef_horsepower:.4f} - {direction} relationship")
print(f"🎯 For 98 HP car: {prediction:.1f} mpg predicted")

print(f"\n🚗 Analysis shows that horsepower is a strong predictor of fuel efficiency!")
print(f"📊 Higher horsepower vehicles tend to have lower MPG (fuel efficiency)")