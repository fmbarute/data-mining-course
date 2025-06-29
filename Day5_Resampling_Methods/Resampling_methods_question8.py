import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm


np.random.seed(1)
X = np.sort(np.random.randn(100))
y = X - 2 * X ** 2 + np.random.randn(100)  # True model: y = X - 2X² + ϵ

# Create DataFrame
df = pd.DataFrame({'y': y, 'x': X})

# 2. Data Visualization

plt.figure(figsize=(10, 6))
ax = sns.scatterplot(x=X, y=y, alpha=0.7, label='Observed Data')
ax.plot(X, X - 2 * X ** 2, color='red', label='True Population Line')
ax.set_xlabel('X')
ax.set_ylabel('y')
ax.set_title('Data and True Relationship')
ax.grid()
ax.legend()
plt.show()


# 3. LOOCV Implementation

def poly_loocv(data, degree):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(data['x'].values.reshape(-1, 1))
    lm = LinearRegression()

    mse = -cross_val_score(lm, X_poly, data['y'],
                           scoring='neg_mean_squared_error',
                           cv=len(data)).mean()
    return mse


# Compute LOOCV errors
degrees = range(1, 5)
loocv_results = [poly_loocv(df, deg) for deg in degrees]

# Plot LOOCV results
plt.figure(figsize=(8, 5))
plt.plot(degrees, loocv_results, marker='o')
plt.xlabel('Polynomial Degree')
plt.ylabel('LOOCV Error (MSE)')
plt.title('LOOCV Error vs. Polynomial Degree')
plt.grid()
plt.show()


# 4. Statistical Significance

# (Add polynomial terms to DataFrame)
for deg in range(2, 5):
    df[f'x_{deg}'] = df['x'] ** deg


def stat_significance(df, degree):
    features = ['const'] + [f'x_{d}' for d in range(1, degree + 1)]
    X = sm.add_constant(df[[f'x_{d}' for d in range(1, degree + 1)]])
    y = df['y']
    model = sm.OLS(y, X).fit()

    print(f"\n{'=' * 40}")
    print(f"Degree {degree} Regression Results")
    print('=' * 40)
    print(model.summary().tables[1])
    print(f"AIC: {model.aic:.2f}")


# Test degrees 1-4
for i in range(1, 5):
    stat_significance(df, i)


print("\n\n=== Final Analysis ===")
print(f"Best model by LOOCV: Degree {np.argmin(loocv_results) + 1}")
print("Note: Degree 2 matches the true data-generating process (y = X - 2X² + ϵ)")