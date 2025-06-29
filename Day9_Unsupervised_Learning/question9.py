import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler

# Load the data
USArrests = pd.read_csv('/home/nkubito/Data_Minig_Course/Data/USArrests.csv', index_col=0)

# (a) Hierarchical clustering (complete linkage, Euclidean distance)
hc_complete = linkage(USArrests, method='complete', metric='euclidean')

# Plot dendrogram (unscaled data)
plt.figure(figsize=(15, 8))
dendrogram(hc_complete, labels=USArrests.index, leaf_font_size=14)
plt.title('Hierarchical Clustering (Unscaled Data)')
plt.xlabel('States')
plt.ylabel('Euclidean Distance')
plt.show()

# (b) Cut dendrogram for 3 clusters
plt.figure(figsize=(15, 8))
dendrogram(hc_complete, labels=USArrests.index, leaf_font_size=14, truncate_mode='lastp', p=12)
plt.axhline(y=110, color='red', linestyle='--', label='Cut at height=110')
plt.title('Dendrogram with 3 Clusters (Unscaled Data)')
plt.legend()
plt.show()

# Assign clusters
clusters_unscaled = fcluster(hc_complete, t=3, criterion='maxclust')
cluster_assignment_unscaled = pd.DataFrame({'State': USArrests.index, 'Cluster': clusters_unscaled})
print("Cluster Assignments (Unscaled Data):\n", cluster_assignment_unscaled.sort_values('Cluster'))

# (c) Repeat with scaled data (std=1)
scaler = StandardScaler()
USArrests_scaled = scaler.fit_transform(USArrests)
hc_complete_scaled = linkage(USArrests_scaled, method='complete', metric='euclidean')

# Plot dendrogram (scaled data)
plt.figure(figsize=(15, 8))
dendrogram(hc_complete_scaled, labels=USArrests.index, leaf_font_size=14)
plt.title('Hierarchical Clustering (Scaled Data)')
plt.xlabel('States')
plt.ylabel('Euclidean Distance')
plt.show()

# (d) Compare effects of scaling
clusters_scaled = fcluster(hc_complete_scaled, t=3, criterion='maxclust')
cluster_assignment_scaled = pd.DataFrame({'State': USArrests.index, 'Cluster': clusters_scaled})
print("\nCluster Assignments (Scaled Data):\n", cluster_assignment_scaled.sort_values('Cluster'))

# Justification for scaling
print("\nJustification for Scaling:")
print("""
1. **Effect of Scaling**: 
   - Unscaled data gives more weight to variables with larger magnitudes (e.g., 'Assault' dominates).
   - Scaling ensures equal contribution from all variables (mean=0, std=1).

2. **Should Variables Be Scaled?** 
   - **Yes**, if variables are on different scales (e.g., 'Murder' vs. 'UrbanPop').
   - Scaling avoids bias toward high-magnitude variables and reflects true dissimilarity.
""")