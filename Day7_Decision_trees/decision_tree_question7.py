import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load Boston dataset
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = boston.target

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Parameter ranges to evaluate
max_features_range = [2, 4, 6, 8, 10, 12]  # From sqrt(p)=3.6 to all features
n_estimators_range = [10, 50, 100, 200, 300, 400, 500]

# Store results
results = []

# Evaluate all combinations
for max_features in max_features_range:
    for n_estimators in n_estimators_range:
        rf = RandomForestRegressor(
            max_features=max_features,
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        test_mse = mean_squared_error(y_test, rf.predict(X_test))
        results.append({
            'max_features': max_features,
            'n_estimators': n_estimators,
            'test_mse': test_mse
        })

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Pivot for heatmap
heatmap_data = results_df.pivot("n_estimators", "max_features", "test_mse")

# Create plot
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlGnBu_r",
            cbar_kws={'label': 'Test MSE'})
plt.title("Random Forest Performance on Boston Housing Data")
plt.xlabel("Number of Features Considered at Each Split (max_features)")
plt.ylabel("Number of Trees (n_estimators)")
plt.show()

# Additional line plot for better interpretation
plt.figure(figsize=(12, 6))
for max_features in max_features_range:
    subset = results_df[results_df['max_features'] == max_features]
    plt.plot(subset['n_estimators'], subset['test_mse'],
             label=f'max_features={max_features}')

plt.title("Random Forest Test MSE by Number of Trees")
plt.xlabel("Number of Trees (n_estimators)")
plt.ylabel("Test MSE")
plt.legend()
plt.grid(True)
plt.show()