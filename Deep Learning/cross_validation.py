import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, make_scorer
import matplotlib.pyplot as plt


# Alternative data loading (if ISLP not available)
def load_auto_data():
    """Load Auto dataset from UCI repository if ISLP not available"""
    try:
        from ISLP import load_data
        return load_data('Auto')
    except ImportError:
        # Fallback to UCI repository
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
        column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
                        'acceleration', 'model_year', 'origin', 'car_name']
        auto_data = pd.read_csv(url, delim_whitespace=True, names=column_names, na_values='?')
        auto_data = auto_data.dropna().drop('car_name', axis=1)
        return auto_data


# Load the data
Auto = load_auto_data()
print(f"Dataset shape: {Auto.shape}")
print(f"Columns: {Auto.columns.tolist()}")

# =============================================================================
# 1. VALIDATION SET APPROACH (Original approach, corrected)
# =============================================================================
print("\n" + "=" * 60)
print("1. VALIDATION SET APPROACH")
print("=" * 60)

# Split data into training and validation sets
Auto_train, Auto_valid = train_test_split(Auto,
                                          test_size=196,
                                          random_state=0)

print(f"Training set size: {len(Auto_train)}")
print(f"Validation set size: {len(Auto_valid)}")


def evalMSE_statsmodels(terms_degree, response, train, test):
    """
    Evaluate MSE using statsmodels with polynomial features
    """
    # Create polynomial features
    X_train = train['horsepower'].values.reshape(-1, 1)
    X_test = test['horsepower'].values.reshape(-1, 1)

    # Create polynomial features
    poly_features = PolynomialFeatures(degree=terms_degree, include_bias=False)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)

    # Add constant term for statsmodels
    X_train_poly = sm.add_constant(X_train_poly)
    X_test_poly = sm.add_constant(X_test_poly)

    y_train = train[response].values
    y_test = test[response].values

    # Fit model using statsmodels
    model = sm.OLS(y_train, X_train_poly)
    results = model.fit()

    # Predict and calculate MSE
    test_pred = results.predict(X_test_poly)
    mse = np.mean((y_test - test_pred) ** 2)

    return mse, results


# Evaluate different polynomial degrees
print("\nValidation Set Results:")
MSE_validation = np.zeros(5)  # Test degrees 1-5
models_validation = []

for idx, degree in enumerate(range(1, 6)):
    mse, model = evalMSE_statsmodels(degree, 'mpg', Auto_train, Auto_valid)
    MSE_validation[idx] = mse
    models_validation.append(model)
    print(f"Degree {degree}: MSE = {mse:.2f}")

# Try different random splits to show variability
print("\nValidation Set with Different Random Seeds:")
random_seeds = [0, 1, 2, 3, 42]
validation_results = []

for seed in random_seeds:
    Auto_train_temp, Auto_valid_temp = train_test_split(Auto,
                                                        test_size=196,
                                                        random_state=seed)
    mse_temp = np.zeros(5)
    for idx, degree in enumerate(range(1, 6)):
        mse_temp[idx], _ = evalMSE_statsmodels(degree, 'mpg', Auto_train_temp, Auto_valid_temp)
    validation_results.append(mse_temp)
    print(f"Seed {seed}: Best degree = {np.argmin(mse_temp) + 1}, MSE = {np.min(mse_temp):.2f}")

validation_results = np.array(validation_results)

# =============================================================================
# 2. K-FOLD CROSS-VALIDATION
# =============================================================================
print("\n" + "=" * 60)
print("2. K-FOLD CROSS-VALIDATION")
print("=" * 60)


def polynomial_cv_sklearn(X, y, max_degree=5, cv_folds=5, random_state=42):
    """
    Perform k-fold cross-validation for polynomial regression using sklearn
    """
    cv_results = []

    for degree in range(1, max_degree + 1):
        # Create pipeline with polynomial features and linear regression
        poly_model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('linear', LinearRegression())
        ])

        # Perform cross-validation
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

        # Use negative MSE as scoring (sklearn convention)
        scores = cross_val_score(poly_model, X, y, cv=kfold,
                                 scoring='neg_mean_squared_error')

        # Convert back to positive MSE
        mse_scores = -scores
        mean_mse = np.mean(mse_scores)
        std_mse = np.std(mse_scores)

        cv_results.append({
            'degree': degree,
            'cv_scores': mse_scores,
            'mean_mse': mean_mse,
            'std_mse': std_mse
        })

        print(f"Degree {degree}: CV MSE = {mean_mse:.2f} ± {std_mse:.2f}")

    return cv_results


# Prepare data for cross-validation
X = Auto['horsepower'].values.reshape(-1, 1)
y = Auto['mpg'].values

# Perform 5-fold cross-validation
print("5-Fold Cross-Validation Results:")
cv_results_5fold = polynomial_cv_sklearn(X, y, max_degree=5, cv_folds=5)

# Perform 10-fold cross-validation for comparison
print("\n10-Fold Cross-Validation Results:")
cv_results_10fold = polynomial_cv_sklearn(X, y, max_degree=5, cv_folds=10)

# =============================================================================
# 3. LEAVE-ONE-OUT CROSS-VALIDATION (LOOCV)
# =============================================================================
print("\n" + "=" * 60)
print("3. LEAVE-ONE-OUT CROSS-VALIDATION")
print("=" * 60)


def loocv_polynomial(X, y, max_degree=5):
    """
    Perform Leave-One-Out Cross-Validation
    Note: This can be slow for large datasets
    """
    n_samples = len(X)
    loocv_results = []

    for degree in range(1, max_degree + 1):
        poly_model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('linear', LinearRegression())
        ])

        # LOOCV is equivalent to KFold with n_splits = n_samples
        loocv = KFold(n_splits=n_samples, shuffle=False)

        scores = cross_val_score(poly_model, X, y, cv=loocv,
                                 scoring='neg_mean_squared_error')

        mse_scores = -scores
        mean_mse = np.mean(mse_scores)

        loocv_results.append({
            'degree': degree,
            'mean_mse': mean_mse,
            'n_folds': n_samples
        })

        print(f"Degree {degree}: LOOCV MSE = {mean_mse:.2f}")

    return loocv_results


# Perform LOOCV (might be slow for large datasets)
if len(Auto) <= 500:  # Only run LOOCV if dataset is not too large
    print("LOOCV Results:")
    loocv_results = loocv_polynomial(X, y, max_degree=5)
else:
    print("Dataset too large for LOOCV demonstration. Skipping...")
    loocv_results = None

# =============================================================================
# 4. VISUALIZATION AND COMPARISON
# =============================================================================
print("\n" + "=" * 60)
print("4. RESULTS COMPARISON")
print("=" * 60)

# Create comparison plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

degrees = range(1, 6)

# Plot 1: Validation Set Variability
ax1.plot(degrees, MSE_validation, 'bo-', linewidth=2, markersize=8, label='Single Split')
for i, seed_results in enumerate(validation_results):
    alpha = 0.3 if i > 0 else 0.7
    label = f'Seed {random_seeds[i]}' if i < 3 else None
    ax1.plot(degrees, seed_results, 'o-', alpha=alpha, label=label)

ax1.set_xlabel('Polynomial Degree')
ax1.set_ylabel('MSE')
ax1.set_title('Validation Set Approach (Different Random Seeds)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Cross-Validation Results
cv_means_5fold = [result['mean_mse'] for result in cv_results_5fold]
cv_stds_5fold = [result['std_mse'] for result in cv_results_5fold]

cv_means_10fold = [result['mean_mse'] for result in cv_results_10fold]
cv_stds_10fold = [result['std_mse'] for result in cv_results_10fold]

ax2.errorbar(degrees, cv_means_5fold, yerr=cv_stds_5fold,
             marker='o', label='5-Fold CV', capsize=5, linewidth=2)
ax2.errorbar(degrees, cv_means_10fold, yerr=cv_stds_10fold,
             marker='s', label='10-Fold CV', capsize=5, linewidth=2)

if loocv_results:
    loocv_means = [result['mean_mse'] for result in loocv_results]
    ax2.plot(degrees, loocv_means, '^-', label='LOOCV', linewidth=2, markersize=8)

ax2.set_xlabel('Polynomial Degree')
ax2.set_ylabel('MSE')
ax2.set_title('Cross-Validation Methods Comparison')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: CV Score Distributions (Boxplot)
cv_score_data = [result['cv_scores'] for result in cv_results_5fold]
positions = degrees
ax3.boxplot(cv_score_data, positions=positions, widths=0.6)
ax3.set_xlabel('Polynomial Degree')
ax3.set_ylabel('MSE')
ax3.set_title('Distribution of 5-Fold CV Scores')
ax3.grid(True, alpha=0.3)

# Plot 4: Best Models Fitted to Data
best_degree_cv = degrees[np.argmin(cv_means_5fold)]
print(f"\nBest polynomial degree (5-fold CV): {best_degree_cv}")

# Fit best model and plot
best_model = Pipeline([
    ('poly', PolynomialFeatures(degree=best_degree_cv)),
    ('linear', LinearRegression())
])
best_model.fit(X, y)

X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_plot = best_model.predict(X_plot)

ax4.scatter(X, y, alpha=0.6, color='lightblue', label='Data')
ax4.plot(X_plot, y_plot, 'r-', linewidth=2,
         label=f'Best Model (Degree {best_degree_cv})')
ax4.set_xlabel('Horsepower')
ax4.set_ylabel('MPG')
ax4.set_title(f'Best Polynomial Model (Degree {best_degree_cv})')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =============================================================================
# 5. SUMMARY TABLE
# =============================================================================
print("\nSUMMARY TABLE:")
print("=" * 80)
summary_data = []

for degree in degrees:
    row = {'Degree': degree}

    # Validation set (mean across different seeds)
    val_mse_mean = np.mean([result[degree - 1] for result in validation_results])
    val_mse_std = np.std([result[degree - 1] for result in validation_results])
    row['Validation_MSE'] = f"{val_mse_mean:.2f} ± {val_mse_std:.2f}"

    # 5-fold CV
    cv5_result = cv_results_5fold[degree - 1]
    row['5Fold_CV_MSE'] = f"{cv5_result['mean_mse']:.2f} ± {cv5_result['std_mse']:.2f}"

    # 10-fold CV
    cv10_result = cv_results_10fold[degree - 1]
    row['10Fold_CV_MSE'] = f"{cv10_result['mean_mse']:.2f} ± {cv10_result['std_mse']:.2f}"

    # LOOCV
    if loocv_results:
        loocv_result = loocv_results[degree - 1]
        row['LOOCV_MSE'] = f"{loocv_result['mean_mse']:.2f}"
    else:
        row['LOOCV_MSE'] = "N/A"

    summary_data.append(row)

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

print(f"\nRecommended model: Polynomial degree {best_degree_cv} (based on 5-fold CV)")
print(f"Expected test MSE: {cv_means_5fold[best_degree_cv - 1]:.2f} ± {cv_stds_5fold[best_degree_cv - 1]:.2f}")