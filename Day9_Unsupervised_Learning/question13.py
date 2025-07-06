import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt

# (a) Load the data
data = pd.read_csv('/home/nkubito/Data_Minig_Course/Data/Ch12Ex13.csv', header=None)
data = data.T  # Transpose so samples are rows and genes are columns
print(f"Data shape: {data.shape}")  # Should be (40, 1000) - 40 samples, 1000 genes

# (b) Hierarchical clustering with different linkage methods
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
corr_matrix = np.corrcoef(data_scaled)
corD = 1 - corr_matrix  # Correlation-based dissimilarity

# Create labels for the samples (first 20 healthy, last 20 diseased)
labels = ['H'] * 20 + ['D'] * 20  # Using shorter labels for better visibility

# Create three separate figures instead of subplots for better visibility
linkage_methods = ['complete', 'single', 'average']
for method in linkage_methods:
    plt.figure(figsize=(12, 6))
    hc = linkage(corD, method=method, metric='correlation')
    dendrogram(hc,
              labels=labels,
              color_threshold=0.5,
              leaf_rotation=90,  # Rotate labels for better readability
              leaf_font_size=10)  # Adjust font size
    plt.title(f"{method.capitalize()} Linkage", fontsize=14)
    plt.axhline(y=0.5, color='r', linestyle='--')
    plt.ylabel("Dissimilarity")
    plt.tight_layout()  # Adjust layout to prevent label cutoff
    plt.show()

# Get cluster assignments for each method
hc_complete = linkage(corD, method='complete', metric='correlation')
hc_single = linkage(corD, method='single', metric='correlation')
hc_avg = linkage(corD, method='average', metric='correlation')

complete_clusters = fcluster(hc_complete, t=2, criterion='maxclust')
single_clusters = fcluster(hc_single, t=2, criterion='maxclust')
avg_clusters = fcluster(hc_avg, t=2, criterion='maxclust')

print("\nCluster assignments:")
print("Complete linkage:", complete_clusters)
print("Single linkage:", single_clusters)
print("Average linkage:", avg_clusters)

# (c) Identify genes that differ most across groups
from scipy.stats import ttest_ind

healthy_data = data_scaled[:20]  # First 20 samples are healthy
diseased_data = data_scaled[20:]  # Last 20 samples are diseased

t_stats, p_values = ttest_ind(healthy_data, diseased_data, axis=0)

# Get top 10 genes with smallest p-values (most significant differences)
top_genes_idx = np.argsort(p_values)[:10]
print("\nTop 10 genes that differ most between groups:")
print(f"Gene indices: {top_genes_idx}")
print(f"Corresponding p-values: {p_values[top_genes_idx]}")
print(f"T-statistics: {t_stats[top_genes_idx]}")

# Create a DataFrame for better visualization of top genes
top_genes_df = pd.DataFrame({
    'Gene_Index': top_genes_idx,
    'p_value': p_values[top_genes_idx],
    't_statistic': t_stats[top_genes_idx],
    'abs_t_stat': np.abs(t_stats[top_genes_idx])
}).sort_values('p_value')

print("\nTop differentially expressed genes:")
print(top_genes_df)