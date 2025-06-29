import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# (a) Generate simulated data with 3 classes
np.random.seed(0)
data = np.random.normal(size=(60, 50))  # 60 obs, 50 variables
data[:20, :] += 1                       # Class 1: mean shift +1
data[20:40, :] += 0                     # Class 2: no shift (mean=0)
data[40:, :] -= 1                       # Class 3: mean shift -1

# True class labels (0, 1, 2)
true_labels = np.array([0]*20 + [1]*20 + [2]*20)

# (b) Perform PCA and plot first two PCs
pca = PCA()
pca_scores = pca.fit_transform(data)[:, :2]

plt.figure(figsize=(8, 6))
plt.scatter(pca_scores[:20, 0], pca_scores[:20, 1], label='Class 1 (+1 shift)', alpha=0.7)
plt.scatter(pca_scores[20:40, 0], pca_scores[20:40, 1], label='Class 2 (no shift)', alpha=0.7)
plt.scatter(pca_scores[40:, 0], pca_scores[40:, 1], label='Class 3 (-1 shift)', alpha=0.7)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA: First Two Principal Components')
plt.legend()
plt.show()

# (c) K-means with K=3
kmeans3 = KMeans(n_clusters=3, n_init=20, random_state=0)
kmeans3_labels = kmeans3.fit_predict(data)
print("\n(c) K=3 Clustering vs True Labels:")
print(pd.crosstab(kmeans3_labels, true_labels, rownames=['Cluster'], colnames=['True Class']))

# (d) K-means with K=2
kmeans2 = KMeans(n_clusters=2, n_init=20, random_state=0)
kmeans2_labels = kmeans2.fit_predict(data)
print("\n(d) K=2 Clustering vs True Labels:")
print(pd.crosstab(kmeans2_labels, true_labels, rownames=['Cluster'], colnames=['True Class']))

# (e) K-means with K=4
kmeans4 = KMeans(n_clusters=4, n_init=20, random_state=0)
kmeans4_labels = kmeans4.fit_predict(data)
print("\n(e) K=4 Clustering vs True Labels:")
print(pd.crosstab(kmeans4_labels, true_labels, rownames=['Cluster'], colnames=['True Class']))

# (f) K-means on PCA scores (K=3)
kmeans_pca = KMeans(n_clusters=3, n_init=20, random_state=0)
kmeans_pca_labels = kmeans_pca.fit_predict(pca_scores)
print("\n(f) K=3 on PCA Scores vs True Labels:")
print(pd.crosstab(kmeans_pca_labels, true_labels, rownames=['Cluster'], colnames=['True Class']))

# (g) K-means on scaled data (K=3)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
kmeans_scaled = KMeans(n_clusters=3, n_init=20, random_state=0)
kmeans_scaled_labels = kmeans_scaled.fit_predict(data_scaled)
print("\n(g) K=3 on Scaled Data vs True Labels:")
print(pd.crosstab(kmeans_scaled_labels, true_labels, rownames=['Cluster'], colnames=['True Class']))