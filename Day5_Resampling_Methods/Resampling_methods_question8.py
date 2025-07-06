# Import necessary libraries
import numpy as np               # For numerical operations and array handling
import pandas as pd              # For data manipulation and DataFrame operations
import matplotlib.pyplot as plt  # For basic plotting
import seaborn as sns            # For enhanced visualizations
from sklearn.preprocessing import PolynomialFeatures  # For generating polynomial features
from sklearn.linear_model import LinearRegression    # For linear regression modeling
from sklearn.model_selection import cross_val_score  # For cross-validation
import statsmodels.api as sm     # For statistical modeling and tests

# Set random seed for reproducibility
np.random.seed(1)  # Ensures the same random numbers are generated each time

# Generate synthetic data
X = np.sort(np.random.randn(100))  # Create 100 random normal values and sort them
y = X - 2 * X ** 2 + np.random.randn(100)  # Create y using quadratic relationship with noise

# Create DataFrame and add polynomial terms
df = pd.DataFrame({'y': y, 'x_1': X})  # Initialize DataFrame with y and x_1 (linear term)
for deg in range(2, 5):                # Loop through degrees 2 to 4
    df[f'x_{deg}'] = df['x_1'] ** deg  # Add squared, cubic, and quartic terms

# Visualize the data and true relationship
plt.figure(figsize=(10, 6))  # Create figure with specified size
# Create scatter plot of observed data
ax = sns.scatterplot(x='x_1', y='y', data=df, alpha=0.7, label='Observed Data')
# Plot the true underlying relationship (without noise)
ax.plot(df['x_1'], df['x_1'] - 2 * df['x_1'] ** 2, color='red', label='True Population Line')
ax.set_xlabel('X')           # Set x-axis label
ax.set_ylabel('y')           # Set y-axis label
ax.set_title('Data and True Relationship')  # Set plot title
ax.grid()                    # Add grid lines
ax.legend()                  # Show legend
plt.show()                   # Display the plot

# Define LOOCV function for polynomial regression
def poly_loocv(data, degree):
    # Create polynomial features up to specified degree
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    # Transform the base feature (x_1) into polynomial features
    X_poly = poly.fit_transform(data['x_1'].values.reshape(-1, 1))
    # Initialize linear regression model
    lm = LinearRegression()
    # Perform LOOCV (since cv=len(data)) and calculate mean squared error
    mse = -cross_val_score(lm, X_poly, data['y'],
                         scoring='neg_mean_squared_error',
                         cv=len(data)).mean()
    return mse  # Return the average MSE across all folds

# Calculate LOOCV errors for polynomial degrees 1 through 4
degrees = range(1, 5)  # Degrees to evaluate (1=linear, 2=quadratic, etc.)
# Compute LOOCV MSE for each degree using list comprehension
loocv_results = [poly_loocv(df, deg) for deg in degrees]

# Plot LOOCV results
plt.figure(figsize=(8, 5))  # Create new figure
plt.plot(degrees, loocv_results, marker='o')  # Plot MSE vs. polynomial degree
plt.xlabel('Polynomial Degree')  # Label x-axis
plt.ylabel('LOOCV Error (MSE)')  # Label y-axis
plt.title('LOOCV Error vs. Polynomial Degree')  # Add title
plt.grid()  # Add grid lines
plt.show()  # Display plot

# Define function to analyze statistical significance
def stat_significance(df, degree):
    # Create list of feature names including constant term
    features = ['const'] + [f'x_{d}' for d in range(1, degree + 1)]
    # Prepare design matrix with specified polynomial terms
    X = sm.add_constant(df[[f'x_{d}' for d in range(1, degree + 1)]])
    y = df['y']  # Get response variable
    # Fit ordinary least squares regression model
    model = sm.OLS(y, X).fit()

    # Print results header
    print(f"\n{'=' * 40}")
    print(f"Degree {degree} Regression Results")
    print('=' * 40)
    # Print coefficients table (shows estimates, std errors, p-values)
    print(model.summary().tables[1])
    # Print Akaike Information Criterion (lower is better)
    print(f"AIC: {model.aic:.2f}")

# Analyze statistical significance for each polynomial degree
for i in range(1, 5):  # Loop through degrees 1 to 4
    stat_significance(df, i)  # Call significance testing function

# Print final conclusions
print("\n\n=== Final Analysis ===")
# Find degree with minimum LOOCV error (adding 1 because Python is 0-indexed)
print(f"Best model by LOOCV: Degree {np.argmin(loocv_results) + 1}")
# Note that this matches the true data-generating process
print("Note: Degree 2 matches the true data-generating process (y = X - 2X² + ϵ)")