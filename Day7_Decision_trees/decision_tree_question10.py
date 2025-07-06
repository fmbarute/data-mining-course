import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from ISLP import load_data

# (a) Load and preprocess data
hitters = load_data('Hitters').dropna()

# Verify the column names in the dataset
print("Available columns:", hitters.columns.tolist())

# Select features - using actual column names from the dataset
X = hitters.drop(['Salary'], axis=1)  # Only drop Salary, keep other columns
y = np.log(hitters['Salary'])  # Log-transform

# Convert categorical variables - using correct column names
categorical_cols = ['League', 'Division', 'NewLeague']
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# (b) Create train/test sets
X_train = X.iloc[:200]
X_test = X.iloc[200:]
y_train = y.iloc[:200]
y_test = y.iloc[200:]

# Standardize features
scaler = StandardScaler()
numeric_cols = X.select_dtypes(include=[np.number]).columns
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# (c)-(d) Boosting with different shrinkage parameters
shrinkage_values = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
train_mses = []
test_mses = []

for shrinkage in shrinkage_values:
    boost = GradientBoostingRegressor(n_estimators=1000,
                                      learning_rate=shrinkage,
                                      random_state=42)
    boost.fit(X_train, y_train)

    train_pred = boost.predict(X_train)
    train_mse = mean_squared_error(y_train, train_pred)
    train_mses.append(train_mse)

    test_pred = boost.predict(X_test)
    test_mse = mean_squared_error(y_test, test_pred)
    test_mses.append(test_mse)

# Plot training MSE
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(shrinkage_values, train_mses, 'bo-')
plt.xscale('log')
plt.xlabel('Shrinkage (λ)')
plt.ylabel('Training MSE')
plt.title('Training MSE vs Shrinkage')

# Plot test MSE
plt.subplot(1, 2, 2)
plt.plot(shrinkage_values, test_mses, 'ro-')
plt.xscale('log')
plt.xlabel('Shrinkage (λ)')
plt.ylabel('Test MSE')
plt.title('Test MSE vs Shrinkage')
plt.tight_layout()
plt.show()

# (e) Compare with other methods
# Linear Regression
lm = LinearRegression()
lm.fit(X_train, y_train)
lm_mse = mean_squared_error(y_test, lm.predict(X_test))

# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_mse = mean_squared_error(y_test, ridge.predict(X_test))

print("\n(e) Comparison of Test MSE:")
print(f"Boosting (best): {min(test_mses):.4f}")
print(f"Linear Regression: {lm_mse:.4f}")
print(f"Ridge Regression: {ridge_mse:.4f}")

# (f) Variable importance
best_boost = GradientBoostingRegressor(n_estimators=1000,
                                       learning_rate=shrinkage_values[np.argmin(test_mses)],
                                       random_state=42)
best_boost.fit(X_train, y_train)

importance = pd.Series(best_boost.feature_importances_, index=X.columns)
print("\n(f) Top 5 Important Features:")
print(importance.sort_values(ascending=False).head(5))

# (g) Bagging
bag = RandomForestRegressor(n_estimators=1000,
                            max_features=X.shape[1],
                            random_state=42)
bag.fit(X_train, y_train)
bag_mse = mean_squared_error(y_test, bag.predict(X_test))
print(f"\n(g) Bagging Test MSE: {bag_mse:.4f}")