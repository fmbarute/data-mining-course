import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load and standardize the data
USArrests = pd.read_csv('/home/nkubito/Data_Minig_Course/Data/USArrests.csv', index_col=0)
scaler = StandardScaler()
USArrests_scaled = scaler.fit_transform(USArrests)  # Center and scale

# (a) Using PCA's explained_variance_ratio_
pca = PCA()
pca.fit(USArrests_scaled)
pves_method_a = pca.explained_variance_ratio_
print("PVE (Method A - explained_variance_ratio_):\n", pves_method_a)

# (b) Using Equation 12.10 directly
n, p = USArrests_scaled.shape
loadings = pca.components_.T  # Transpose loadings (each column is a PC)

# Calculate PVE for each principal component
pves_method_b = []
for k in range(p):
    variance_explained = 0
    for i in range(n):
        z_i = USArrests_scaled[i, :]  # Standardized observation i
        phi_k = loadings[:, k]         # Loadings for PC k
        variance_explained += (np.dot(z_i, phi_k)) ** 2
    total_variance = np.sum(USArrests_scaled ** 2)  # Sum of squares of all elements
    pves_method_b.append(variance_explained / total_variance)

print("\nPVE (Method B - Equation 12.10):\n", np.array(pves_method_b))

# Verify equality (up to numerical precision)
assert np.allclose(pves_method_a, pves_method_b), "Results differ!"
print("\nVerification: Both methods yield the same PVE values.")