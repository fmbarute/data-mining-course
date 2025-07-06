import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline


# Load and prepare the data
def load_auto_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
                    'acceleration', 'model_year', 'origin', 'car_name']
    auto_data = pd.read_csv(url, delim_whitespace=True, names=column_names, na_values='?')
    auto_data = auto_data.dropna().drop('car_name', axis=1)
    return auto_data


# Prepare data
auto_data = load_auto_data()
X = auto_data['horsepower'].values.reshape(-1, 1)
y = auto_data['mpg'].values

# Split data into training (70%) and validation (30%) sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)


# Function to evaluate polynomial models
def evaluate_models(max_degree=5):
    results = []

    for degree in range(1, max_degree + 1):
        # Create polynomial regression model
        model = make_pipeline(
            PolynomialFeatures(degree=degree),
            LinearRegression()
        )

        # Fit model on training data
        model.fit(X_train, y_train)

        # Predict on training and validation sets
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        # Calculate MSE
        train_mse = mean_squared_error(y_train, y_train_pred)
        val_mse = mean_squared_error(y_val, y_val_pred)

        results.append({
            'Degree': degree,
            'Train MSE': train_mse,
            'Validation MSE': val_mse,
            'Model': model
        })

        print(f"Degree {degree}:")
        print(f"  Training MSE: {train_mse:.2f}")
        print(f"  Validation MSE: {val_mse:.2f}")
        print("-" * 40)

    return pd.DataFrame(results)


# Evaluate models up to 5th degree polynomial
results_df = evaluate_models(max_degree=5)

# Find the best model based on validation MSE
best_model_info = results_df.loc[results_df['Validation MSE'].idxmin()]
print("\nBest Model:")
print(f"Polynomial Degree: {best_model_info['Degree']}")
print(f"Validation MSE: {best_model_info['Validation MSE']:.2f}")

# Plot MSE vs polynomial degree
plt.figure(figsize=(10, 6))
plt.plot(results_df['Degree'], results_df['Train MSE'], 'bo-', label='Training MSE')
plt.plot(results_df['Degree'], results_df['Validation MSE'], 'ro-', label='Validation MSE')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Model Complexity vs Error')
plt.xticks(results_df['Degree'])
plt.legend()
plt.grid(True)
plt.show()

# Visualize the best model's predictions
best_model = best_model_info['Model']

# Create range for plotting
x_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_plot = best_model.predict(x_plot)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='gray', alpha=0.5, label='Actual Data')
plt.plot(x_plot, y_plot, color='red', linewidth=2,
         label=f'Degree {best_model_info["Degree"]} Polynomial Fit')
plt.xlabel('Horsepower')
plt.ylabel('MPG')
plt.title(f'Best Polynomial Model (Degree {best_model_info["Degree"]})')
plt.legend()
plt.grid(True)
plt.show()

# Print model coefficients for interpretation
if best_model_info['Degree'] == 1:
    print("\nLinear Model Coefficients:")
    print(f"Intercept: {best_model.named_steps['linearregression'].intercept_:.2f}")
    print(f"Coefficient: {best_model.named_steps['linearregression'].coef_[1]:.2f}")
else:
    print("\nPolynomial Model Coefficients:")
    print("Intercept:", best_model.named_steps['linearregression'].intercept_)
    print("Coefficients:", best_model.named_steps['linearregression'].coef_)