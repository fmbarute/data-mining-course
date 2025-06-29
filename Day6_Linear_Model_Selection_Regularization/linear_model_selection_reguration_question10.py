import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from itertools import combinations
from tqdm import tqdm

# (a) Generate simulated data
np.random.seed(42)
n = 1000
p = 20

# Create true beta with some zeros (sparse true model)
true_beta = np.zeros(p)
true_beta[[0, 3, 7, 12, 18]] = [1.5, -2.0, 1.0, 3.0, -1.5]  # Only 5 non-zero coefficients

# Generate X matrix
X = np.random.normal(size=(n, p))

# Generate response Y with noise
epsilon = np.random.normal(scale=2, size=n)
Y = X @ true_beta + epsilon

# (b) Split into training (100) and test (900)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=900, random_state=42)


# (c)-(d) Best subset selection
def best_subset_selection(X_train, y_train, X_test, y_test, max_size=20):
    n_features = X_train.shape[1]
    results = []

    for k in tqdm(range(1, max_size + 1)):
        best_score_train = np.inf
        best_score_test = np.inf
        best_subset = None
        best_coefs = None

        for subset in combinations(range(n_features), k):
            X_subset = X_train[:, subset]
            model = LinearRegression().fit(X_subset, y_train)

            # Training MSE
            train_pred = model.predict(X_subset)
            train_mse = mean_squared_error(y_train, train_pred)

            # Test MSE
            test_subset = X_test[:, subset]
            test_pred = model.predict(test_subset)
            test_mse = mean_squared_error(y_test, test_pred)

            if train_mse < best_score_train:
                best_score_train = train_mse
                best_score_test = test_mse
                best_subset = subset
                best_coefs = np.zeros(n_features)
                best_coefs[list(subset)] = model.coef_
                best_intercept = model.intercept_

        results.append({
            'size': k,
            'train_mse': best_score_train,
            'test_mse': best_score_test,
            'subset': best_subset,
            'coefs': best_coefs,
            'intercept': best_intercept
        })

    return pd.DataFrame(results)


results = best_subset_selection(X_train, y_train, X_test, y_test)

# (c) Plot training MSE
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(results['size'], results['train_mse'], 'bo-')
plt.xlabel('Number of predictors')
plt.ylabel('Training MSE')
plt.title('Training MSE vs Model Size')

# (d) Plot test MSE
plt.subplot(1, 2, 2)
plt.plot(results['size'], results['test_mse'], 'ro-')
plt.xlabel('Number of predictors')
plt.ylabel('Test MSE')
plt.title('Test MSE vs Model Size')
plt.tight_layout()
plt.show()

# (e) Find optimal model size
optimal_size = results.loc[results['test_mse'].idxmin(), 'size']
print(f"\n(e) Optimal model size (minimum test MSE): {optimal_size}")

# (f) Compare with true model
optimal_model = results.iloc[optimal_size - 1]
print("\n(f) Comparison with true model:")
print(f"True non-zero coefficients: {np.where(true_beta != 0)[0]}")
print(f"Selected features in optimal model: {optimal_model['subset']}")

# Calculate coefficient estimation error
coef_errors = []
for r in range(1, 21):
    beta_hat = results.iloc[r - 1]['coefs']
    error = np.sqrt(np.sum((true_beta - beta_hat) ** 2))
    coef_errors.append(error)

# (g) Plot coefficient estimation error
plt.figure(figsize=(10, 6))
plt.plot(range(1, 21), coef_errors, 'go-')
plt.xlabel('Number of predictors')
plt.ylabel(r'$\sqrt{\sum(\beta_j - \hat{\beta}_j)^2}$')
plt.title('Coefficient Estimation Error vs Model Size')
plt.axvline(optimal_size, color='red', linestyle='--')
plt.show()