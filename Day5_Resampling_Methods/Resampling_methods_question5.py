import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data (using correct URL)
try:
    # First try local file
    default = pd.read_csv('Default.csv')
except FileNotFoundError:
    print("Local Default.csv not found - loading from GitHub...")
    try:
        url = "https://raw.githubusercontent.com/StatLearning/StatLearning/master/data/Default.csv"
        default = pd.read_csv(url)
        # Save it locally for future use
        default.to_csv('Default.csv', index=False)
        print("Downloaded and saved Default.csv locally")
    except Exception as e:
        print(f"Failed to download Default.csv: {e}")
        print("Please manually download Default.csv and place it in your working directory")
        print("You can get it from: https://github.com/StatLearning/StatLearning/tree/master/data")
        exit()

# Preprocess data
default['default'] = (default['default'] == 'Yes').astype(int)
default['student'] = (default['student'] == 'Yes').astype(int)

# Rest of your code continues...
X = default[['income', 'balance']]
y = default['default']

# (b) Validation set approach
np.random.seed(1)
for i in range(3):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    model = LogisticRegression().fit(X_train, y_train)
    pred = model.predict(X_val)
    error = 1 - accuracy_score(y_val, pred)
    print(f"Split {i+1} validation error: {error:.4f}")

# (d) Include student dummy
X = default[['income', 'balance', 'student']]
for i in range(3):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    model = LogisticRegression().fit(X_train, y_train)
    pred = model.predict(X_val)
    error = 1 - accuracy_score(y_val, pred)
    print(f"With student - Split {i+1} validation error: {error:.4f}")