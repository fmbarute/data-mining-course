import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from statsmodels.formula.api import glm


# Load and prepare the data (self-contained function)
def load_auto_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
                    'acceleration', 'model_year', 'origin', 'car_name']
    auto_data = pd.read_csv(url, delim_whitespace=True, names=column_names, na_values='?')
    auto_data = auto_data.dropna().drop('car_name', axis=1)
    return auto_data


# Load data
auto_data = load_auto_data()

# Split data (keeping as DataFrame for statsmodels formula interface)
train_data, val_data = train_test_split(auto_data, test_size=0.3, random_state=42)


def evaluate_glm_models(max_degree=5):
    results = []

    for degree in range(1, max_degree + 1):
        # Create polynomial terms
        for d in range(2, degree + 1):
            train_data[f'horsepower_{d}'] = train_data['horsepower'] ** d
            val_data[f'horsepower_{d}'] = val_data['horsepower'] ** d

        # Build formula
        formula = 'mpg ~ horsepower' + ''.join([f' + horsepower_{d}' for d in range(2, degree + 1)])

        # Fit GLM (equivalent to linear regression)
        model = glm(formula, data=train_data,
                    family=sm.families.Gaussian()).fit()

        # Predictions
        y_train_pred = model.predict(train_data)
        y_val_pred = model.predict(val_data)

        # Calculate MSE
        train_mse = mean_squared_error(train_data['mpg'], y_train_pred)
        val_mse = mean_squared_error(val_data['mpg'], y_val_pred)

        results.append({
            'Degree': degree,
            'Train MSE': train_mse,
            'Validation MSE': val_mse,
            'Model': model
        })

        print(f"Degree {degree}:")
        print(model.summary())  # GLM summary output
        print(f"  Training MSE: {train_mse:.2f}")
        print(f"  Validation MSE: {val_mse:.2f}")
        print("-" * 40)

        # Remove polynomial terms for next iteration
        for d in range(2, degree + 1):
            del train_data[f'horsepower_{d}']
            del val_data[f'horsepower_{d}']

    return pd.DataFrame(results)


# Run analysis
glm_results = evaluate_glm_models(max_degree=5)

# Find best model
best_model_info = glm_results.loc[glm_results['Validation MSE'].idxmin()]
print("\nBest Model:")
print(f"Polynomial Degree: {best_model_info['Degree']}")
print(f"Validation MSE: {best_model_info['Validation MSE']:.2f}")

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(glm_results['Degree'], glm_results['Train MSE'], 'bo-', label='Training MSE')
plt.plot(glm_results['Degree'], glm_results['Validation MSE'], 'ro-', label='Validation MSE')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('GLM: Model Complexity vs Error')
plt.xticks(glm_results['Degree'])
plt.legend()
plt.grid(True)
plt.show()