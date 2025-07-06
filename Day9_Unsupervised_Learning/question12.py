import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import random
from sklearn.datasets import fetch_california_housing

# Load California housing dataset
california = fetch_california_housing()
california_data = pd.DataFrame(california.data, columns=california.feature_names)


def matrix_completion2(matrix, M=1, max_iter=1000, print_progress=False):
    """
    Matrix completion using PCA

    Parameters:
    - matrix: Input matrix with missing values (np.nan)
    - M: Number of principal components to use
    - max_iter: Maximum number of iterations
    - print_progress: Whether to print progress information

    Returns:
    - Completed matrix
    """
    Xhat = matrix.copy()
    ismiss = np.isnan(Xhat)
    Xbar = np.nanmean(Xhat, axis=0)
    Xhat[ismiss] = np.take(Xbar, np.where(ismiss)[1])  # Mean imputation

    thresh = 1e-7
    rel_err = 1
    count = 0
    mssold = np.mean(Xhat[~ismiss] ** 2)
    mss0 = np.mean(matrix[~ismiss] ** 2)

    def low_rank(X, M=M):
        """Perform low-rank approximation using PCA"""
        # Center the data
        X_centered = X - np.nanmean(X, axis=0)
        # Get only complete cases for PCA fitting
        complete_cases = X_centered[~np.any(np.isnan(X_centered), axis=1)]
        # Fit PCA
        pca = PCA(n_components=M)
        pca.fit(complete_cases)
        # Transform and reconstruct
        scores = pca.transform(X_centered)
        loadings = pca.components_
        return np.dot(scores, loadings) + np.nanmean(X, axis=0)

    while rel_err > thresh and count < max_iter:
        count += 1

        Xapp = low_rank(Xhat, M=M)
        Xhat[ismiss] = Xapp[ismiss]  # Update missing values

        # Calculate reconstruction error
        mss = np.mean(((matrix[~ismiss] - Xhat[~ismiss])) ** 2)
        rel_err = (mssold - mss) / mss0
        mssold = mss

        if print_progress:
            print(f"Iteration: {count}, MSS: {mss:.3f}, Rel.Err {rel_err:.2e}")

    return Xhat  # Return the completed matrix


# Testing code
percentage = 5

while percentage <= 30:
    print(f'\nPercentage missing - {percentage}%')
    dataset = california_data.copy().values  # Convert to numpy array
    original_dataset = dataset.copy()  # Preserve original data

    # Create random missing values
    num_observations = int(dataset.shape[0] * percentage / 100)
    random_indices = random.sample(range(dataset.shape[0]), num_observations)

    num_columns = int(dataset.shape[1] * percentage / 100) + 1
    selected_columns = random.sample(range(dataset.shape[1]), num_columns)

    ismiss = np.zeros_like(dataset, dtype=bool)
    ismiss[np.ix_(random_indices, selected_columns)] = True

    # Create dataset with missing values
    dataset_missing = dataset.copy()
    dataset_missing[ismiss] = np.nan

    for m in range(1, min(9, dataset.shape[1])):  # Ensure M doesn't exceed num features
        print(f'\nM: {m}', end=' ')
        try:
            X_completed = matrix_completion2(dataset_missing, M=m)

            # Calculate correlation between imputed and true values
            correlation = np.corrcoef(X_completed[ismiss], original_dataset[ismiss])[0, 1]
            print(f'Correlation: {correlation:.4f}')
        except Exception as e:
            print(f'Error with M={m}: {str(e)}')

    print('-' * 60)
    percentage += 5