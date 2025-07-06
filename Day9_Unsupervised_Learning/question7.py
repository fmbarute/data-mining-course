import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

# Load and preprocess the data
USArrests = pd.read_csv('/home/nkubito/Data_Minig_Course/Data/USArrests.csv', index_col=0)
scaler = StandardScaler()
USArrests_scaled = scaler.fit_transform(USArrests)  # Standardize (mean=0, std=1)

# Compute squared Euclidean distances
euc_distances = pairwise_distances(USArrests_scaled, metric='sqeuclidean')

# Compute correlations and convert to distance (1 - r_ij)
corr_matrix = np.corrcoef(USArrests_scaled)  # Correlation matrix
corr_distances = 1 - corr_matrix  # Correlation-based distance

# Flatten the matrices for comparison (excluding diagonal)
n = len(USArrests_scaled)
euc_flat = euc_distances[np.triu_indices(n, k=1)]  # Upper triangle (no diagonal)
corr_flat = corr_distances[np.triu_indices(n, k=1)]

# Check proportionality: Euclidean ≈ k * (1 - r_ij)
# Fit a linear model (without intercept) to find k
k = np.sum(euc_flat * corr_flat) / np.sum(corr_flat**2)
print(f"Proportionality constant (k): {k:.4f}")

# Verify by plotting
import matplotlib.pyplot as plt
plt.scatter(corr_flat, euc_flat, alpha=0.5)
plt.xlabel('1 - Correlation (rij)')
plt.ylabel('Squared Euclidean Distance')
plt.title('Proportionality Check: Euclidean vs. Correlation Distance')
plt.plot([0, 2], [0, 2 * k], 'r--', label=f'Euclidean ≈ {k:.2f} * (1 - r_ij)')
plt.legend()
plt.show()