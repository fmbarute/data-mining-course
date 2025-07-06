import numpy as np
import pandas as pd
import statsmodels.api as sm
import os
from urllib.request import urlretrieve


# Load the dataset with multiple fallback options
def load_default_data():
    # Try multiple possible locations
    possible_urls = [
        "https://raw.githubusercontent.com/StatLearning/StatLearning/master/data/Default.csv",
        "https://www.statlearning.com/s/Default.csv",
        "https://github.com/JWarmenhoven/ISLR-python/raw/master/Notebooks/Data/Default.csv"
    ]

    local_path = 'Default.csv'

    # First try local file
    if os.path.exists(local_path):
        return pd.read_csv(local_path)

    # If not found locally, try each URL
    for url in possible_urls:
        try:
            print(f"Attempting to download from: {url}")
            urlretrieve(url, local_path)
            print("Download successful!")
            return pd.read_csv(local_path)
        except Exception as e:
            print(f"Failed to download from {url}: {str(e)}")
            continue

    # If all URLs fail, create synthetic data
    print("All download attempts failed. Creating synthetic data...")
    np.random.seed(1)
    data = {
        'default': np.random.choice(['Yes', 'No'], 10000, p=[0.033, 0.967]),
        'student': np.random.choice(['Yes', 'No'], 10000, p=[0.285, 0.715]),
        'balance': np.round(np.random.normal(1500, 400, 10000), 2),
        'income': np.round(np.random.normal(45000, 15000, 10000), 2)
    }
    df = pd.DataFrame(data)
    df.to_csv(local_path, index=False)
    return df


# Load the data
default = load_default_data()

# Convert categorical variables to numeric
default['default'] = (default['default'] == 'Yes').astype(int)
default['student'] = (default['student'] == 'Yes').astype(int)

# (a) Standard errors using sm.GLM()
X = sm.add_constant(default[['income', 'balance']])
y = default['default']
model = sm.GLM(y, X, family=sm.families.Binomial()).fit()
print(model.summary())


# (b)-(c) Bootstrap standard errors
def boot_fn(data, index):
    X = sm.add_constant(data.loc[index, ['income', 'balance']])
    y = data.loc[index, 'default']
    model = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    return model.params[1:3]  # Return income and balance coefficients


np.random.seed(1)
boot_coefs = []
for _ in range(1000):
    sample = default.sample(n=len(default), replace=True)
    coefs = boot_fn(sample, sample.index)
    boot_coefs.append(coefs)

boot_coefs = np.array(boot_coefs)
print("\nBootstrap standard errors:")
print(f"Income: {np.std(boot_coefs[:, 0]):.5f}")
print(f"Balance: {np.std(boot_coefs[:, 1]):.5f}")