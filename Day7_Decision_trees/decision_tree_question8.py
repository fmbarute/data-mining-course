import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load Carseats data
url = "https://raw.githubusercontent.com/JWarmenhoven/ISLR-python/master/Notebooks/Data/Carseats.csv"
carseats = pd.read_csv(url)

# Preprocessing: Convert categorical variables to dummy variables
carseats = pd.get_dummies(carseats, columns=['ShelveLoc', 'Urban', 'US'], drop_first=True)

# (a) Split into training and test sets
X = carseats.drop('Sales', axis=1)
y = carseats['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# (b) Fit regression tree
tree = DecisionTreeRegressor(max_depth=3, random_state=42)
tree.fit(X_train, y_train)
tree_pred = tree.predict(X_test)
tree_mse = mean_squared_error(y_test, tree_pred)

plt.figure(figsize=(20, 10))
plot_tree(tree, filled=True, feature_names=X.columns, rounded=True)
plt.show()
print(f"(b) Regression Tree Test MSE: {tree_mse:.2f}")

# (c) Tree pruning with cross-validation
params = {'max_depth': range(1, 11),
          'min_samples_split': range(2, 11),
          'min_samples_leaf': range(1, 6)}

grid = GridSearchCV(DecisionTreeRegressor(random_state=42),
                    params,
                    cv=5,
                    scoring='neg_mean_squared_error')
grid.fit(X_train, y_train)

best_tree = grid.best_estimator_
pruned_pred = best_tree.predict(X_test)
pruned_mse = mean_squared_error(y_test, pruned_pred)

print(f"(c) Optimal parameters: {grid.best_params_}")
print(f"    Pruned Tree Test MSE: {pruned_mse:.2f}")

# (d) Bagging
bag = BaggingRegressor(base_estimator=DecisionTreeRegressor(),
                       n_estimators=500,
                       random_state=42,
                       n_jobs=-1)
bag.fit(X_train, y_train)
bag_pred = bag.predict(X_test)
bag_mse = mean_squared_error(y_test, bag_pred)

# Get feature importances from one of the trees
single_tree = bag.estimators_[0]
importances = pd.Series(single_tree.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(5)

print(f"\n(d) Bagging Test MSE: {bag_mse:.2f}")
print("Top 5 Important Features:")
print(top_features)

# (e) Random Forest
rf_mse_results = []
m_values = range(1, X.shape[1] + 1)

for m in m_values:
    rf = RandomForestRegressor(n_estimators=500,
                               max_features=m,
                               random_state=42,
                               n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_mse_results.append(mean_squared_error(y_test, rf_pred))

    if m == 6:  # Typical default m ≈ p/3
        rf_importances = pd.Series(rf.feature_importances_, index=X.columns)
        rf_top_features = rf_importances.sort_values(ascending=False).head(5)

plt.figure(figsize=(10, 6))
plt.plot(m_values, rf_mse_results, 'bo-')
plt.xlabel('Number of Features Considered at Each Split (m)')
plt.ylabel('Test MSE')
plt.title('Random Forest Performance vs. m')
plt.axvline(x=6, color='red', linestyle='--')
plt.show()

print(f"\n(e) Best Random Forest Test MSE: {min(rf_mse_results):.2f} (m={m_values[np.argmin(rf_mse_results)]})")
print("Top 5 Important Features (m=6):")
print(rf_top_features)

# (f) BART (Bayesian Additive Regression Trees)
try:
    from bartpy.sklearnmodel import SklearnModel

    bart = SklearnModel(n_samples=200, n_burn=50, n_trees=50)
    bart.fit(X_train.values, y_train.values)
    bart_pred = bart.predict(X_test.values)
    bart_mse = mean_squared_error(y_test, bart_pred)
    print(f"\n(f) BART Test MSE: {bart_mse:.2f}")
except ImportError:
    print("\n(f) BART not available (requires bartpy package)")
    print("Install with: pip install bartpy")