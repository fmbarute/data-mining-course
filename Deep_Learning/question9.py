import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Load NYSE data (assuming it has 'Return' column and Date index)
# nyse = pd.read_csv('NYSE.csv', parse_dates=['Date'], index_col='Date')
# For demonstration, let's create synthetic data
np.random.seed(42)
dates = pd.date_range('2000-01-01', '2020-12-31')
nyse = pd.DataFrame({'Return': np.random.normal(0, 1, len(dates))}, index=dates)

# Create lagged features
for lag in range(1, 6):
    nyse[f'Return_lag_{lag}'] = nyse['Return'].shift(lag)

# Add month as a factor
nyse['Month'] = nyse.index.month

# Drop NA rows created by lags
nyse = nyse.dropna()

# Split into features and target
X = nyse.drop('Return', axis=1)
y = nyse['Return']

# Convert month to dummy variables
X = pd.get_dummies(X, columns=['Month'], drop_first=True)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Model without month
model_no_month = LinearRegression()
model_no_month.fit(X_train.drop(X_train.filter(regex='Month').columns, axis=1), y_train)
r2_no_month = r2_score(y_test, model_no_month.predict(X_test.drop(X_test.filter(regex='Month').columns, axis=1)))

# Model with month
model_with_month = LinearRegression()
model_with_month.fit(X_train, y_train)
r2_with_month = r2_score(y_test, model_with_month.predict(X_test))

print(f"R2 without month factor: {r2_no_month:.4f}")
print(f"R2 with month factor: {r2_with_month:.4f}")
print(f"Improvement: {r2_with_month - r2_no_month:.4f}")