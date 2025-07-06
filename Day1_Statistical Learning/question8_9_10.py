# Chapter 2: Statistical Learning - Complete Applied Questions (8, 9, 10)
# Using local CSV files from your Data directory

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Your data path
DATA_PATH = "/home/nkubito/Data_Minig_Course/Data"


def load_local_data(filename):
    """Load dataset from local directory with proper handling"""
    file_path = f"{DATA_PATH}/{filename}"
    print(f"Loading {filename}...")

    try:
        if filename == 'College.csv':
            # College has university names as index
            df = pd.read_csv(file_path, index_col=0)
        elif filename == 'Auto.csv':
            # Auto might have '?' as missing values
            df = pd.read_csv(file_path, na_values=['?'])
            # Remove rows with missing values
            df = df.dropna()
        else:
            df = pd.read_csv(file_path)

        print(f"✓ Loaded {filename} successfully: {df.shape}")
        return df
    except Exception as e:
        print(f"✗ Error loading {filename}: {e}")
        return None


print("=" * 80)
print("CHAPTER 2: STATISTICAL LEARNING - APPLIED EXERCISES")
print("=" * 80)

# =============================================================================
# QUESTION 8: COLLEGE DATASET
# =============================================================================
print("\n" + "=" * 50)
print("QUESTION 8: COLLEGE DATASET ANALYSIS")
print("=" * 50)

college = load_local_data('College.csv')

if college is not None:
    # (a) Data loading - already done
    print("\n(a) ✓ Data loaded successfully")

    # (b) Examine the data
    print(f"\n(b) Data examination:")
    print(f"Shape: {college.shape}")
    print(f"Columns: {list(college.columns)}")
    print(f"Index (college names): {college.index[:3].tolist()}...")

    # (c) Numerical summary
    print(f"\n(c) Summary statistics:")
    print(college.describe().round(2))

    # (d) Scatterplot matrix
    print(f"\n(d) Creating scatterplot matrix...")
    fig = plt.figure(figsize=(12, 8))
    pd.plotting.scatter_matrix(college[['Top10perc', 'Apps', 'Enroll']],
                               alpha=0.6, figsize=(12, 8), diagonal='hist')
    plt.suptitle('College Data: Top10perc, Apps, Enroll')
    plt.tight_layout()
    plt.show()

    # (e) Boxplot
    print(f"\n(e) Boxplot: Outstate vs Private...")
    plt.figure(figsize=(10, 6))
    college.boxplot(column='Outstate', by='Private')
    plt.title('Out-of-State Tuition by College Type')
    plt.suptitle('')  # Remove default title
    plt.show()

    # (f) Elite variable
    print(f"\n(f) Creating Elite variable...")
    college['Elite'] = pd.cut(college['Top10perc'], bins=[0, 50, 100], labels=['No', 'Yes'])
    print(f"Elite universities: {college['Elite'].value_counts()}")

    plt.figure(figsize=(10, 6))
    college.boxplot(column='Outstate', by='Elite')
    plt.title('Out-of-State Tuition by Elite Status')
    plt.suptitle('')
    plt.show()

    # (g) Histograms
    print(f"\n(g) Creating histograms...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    vars_to_plot = ['Apps', 'Accept', 'Enroll', 'Top10perc']
    bins = [50, 30, 25, 20]

    for i, var in enumerate(vars_to_plot):
        row, col = i // 2, i % 2
        axes[row, col].hist(college[var], bins=bins[i], alpha=0.7, edgecolor='black')
        axes[row, col].set_title(f'{var} (bins={bins[i]})')
        axes[row, col].set_xlabel(var)
        axes[row, col].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    # (h) Additional exploration
    print(f"\n(h) Additional exploration:")

    # Most/least expensive colleges
    print("Most expensive (Outstate tuition):")
    print(college.nlargest(5, 'Outstate')[['Private', 'Outstate', 'Top10perc']])

    print("\nHighest graduation rates:")
    print(college.nlargest(5, 'Grad.Rate')[['Private', 'Grad.Rate', 'Top10perc']])

    # Correlation with graduation rate
    numeric_cols = college.select_dtypes(include=[np.number]).columns
    grad_corr = college[numeric_cols].corr()['Grad.Rate'].abs().sort_values(ascending=False)
    print(f"\nVariables most correlated with Grad.Rate:")
    print(grad_corr.head(6))

# =============================================================================
# QUESTION 9: AUTO DATASET
# =============================================================================
print("\n" + "=" * 50)
print("QUESTION 9: AUTO DATASET ANALYSIS")
print("=" * 50)

auto = load_local_data('Auto.csv')

if auto is not None:
    print(f"\n(a) Variable types:")
    print(f"Dataset shape: {auto.shape}")
    print(f"Columns: {list(auto.columns)}")

    # Check data types
    print(f"\nQuantitative variables:")
    quantitative = auto.select_dtypes(include=[np.number]).columns.tolist()
    print(quantitative)

    print(f"\nQualitative variables:")
    qualitative = auto.select_dtypes(exclude=[np.number]).columns.tolist()
    # Note: 'origin' might be coded as numeric but is actually categorical
    if 'origin' in quantitative:
        print("Note: 'origin' should be treated as qualitative despite numeric coding")
        qualitative.append('origin')
        quantitative.remove('origin')
    print(qualitative)

    # (b) Range of quantitative predictors
    print(f"\n(b) Range of quantitative predictors:")
    for var in quantitative:
        if var != 'origin':  # Skip origin as it's categorical
            print(f"{var}: {auto[var].min():.1f} to {auto[var].max():.1f}")

    # (c) Mean and standard deviation
    print(f"\n(c) Mean and standard deviation:")
    stats = auto[quantitative].agg(['mean', 'std'])
    print(stats.round(3))

    # (d) Remove observations 10-85
    print(f"\n(d) After removing observations 10-85:")
    auto_subset = auto.drop(auto.index[9:85])  # 0-based indexing
    print(f"Original size: {len(auto)}, New size: {len(auto_subset)}")

    subset_stats = auto_subset[quantitative].agg(['min', 'max', 'mean', 'std'])
    print(subset_stats.round(3))

    # (e) Graphical investigation
    print(f"\n(e) Creating visualizations...")

    # Correlation matrix
    plt.figure(figsize=(10, 8))
    corr_matrix = auto[quantitative].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Auto Dataset: Correlation Matrix')
    plt.tight_layout()
    plt.show()

    # Key relationships with mpg
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    other_vars = [var for var in quantitative if var != 'mpg']

    for i, var in enumerate(other_vars[:6]):
        row, col = i // 3, i % 3
        axes[row, col].scatter(auto[var], auto['mpg'], alpha=0.6)
        axes[row, col].set_xlabel(var)
        axes[row, col].set_ylabel('mpg')

        # Add correlation
        corr = auto[[var, 'mpg']].corr().iloc[0, 1]
        axes[row, col].set_title(f'mpg vs {var} (r={corr:.3f})')

    plt.tight_layout()
    plt.show()

    # (f) Useful predictors for mpg
    print(f"\n(f) Variables useful for predicting mpg:")
    mpg_corr = auto[quantitative].corr()['mpg'].abs().sort_values(ascending=False)
    print("Correlation with mpg (absolute values):")
    print(mpg_corr)

    print("\nBest predictors for mpg:")
    print("- Weight (strong negative correlation)")
    print("- Displacement (strong negative correlation)")
    print("- Horsepower (strong negative correlation)")
    print("- Year (moderate positive correlation)")

# =============================================================================
# QUESTION 10: BOSTON DATASET
# =============================================================================
print("\n" + "=" * 50)
print("QUESTION 10: BOSTON HOUSING DATASET")
print("=" * 50)

boston = load_local_data('Boston.csv')

if boston is not None:
    # (a) Data loading - already done
    print(f"\n(a) ✓ Boston dataset loaded")

    # (b) Dimensions and meaning
    print(f"\n(b) Dataset dimensions:")
    print(f"Rows: {boston.shape[0]} (Boston suburbs/neighborhoods)")
    print(f"Columns: {boston.shape[1]} (housing characteristics)")
    print(f"Each row represents a Boston suburb with various attributes")

    # (c) Pairwise scatterplots
    print(f"\n(c) Creating pairwise scatterplots...")
    # Select key variables for visualization
    key_vars = ['crim', 'medv', 'lstat', 'rm', 'age', 'dis']

    if all(var in boston.columns for var in key_vars):
        fig = plt.figure(figsize=(15, 12))
        pd.plotting.scatter_matrix(boston[key_vars], alpha=0.6, figsize=(15, 12))
        plt.suptitle('Boston Housing: Key Variables Scatterplot Matrix')
        plt.tight_layout()
        plt.show()
    else:
        print("Some expected columns not found. Available columns:")
        print(boston.columns.tolist())

    # (d) Crime rate associations
    print(f"\n(d) Variables associated with crime rate:")
    if 'crim' in boston.columns:
        crime_corr = boston.corr()['crim'].abs().sort_values(ascending=False)
        print("Correlation with crime rate (absolute values):")
        print(crime_corr.head(8))

        # Visualize top correlations with crime
        top_crime_vars = crime_corr.index[1:4]  # Top 3 (excluding crime itself)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, var in enumerate(top_crime_vars):
            axes[i].scatter(boston[var], boston['crim'], alpha=0.6)
            axes[i].set_xlabel(var)
            axes[i].set_ylabel('Crime Rate')
            axes[i].set_title(f'Crime vs {var}')

        plt.tight_layout()
        plt.show()

    # (e) High values analysis
    print(f"\n(e) Suburbs with extreme values:")
    for var in ['crim', 'tax', 'ptratio']:
        if var in boston.columns:
            q95 = boston[var].quantile(0.95)
            high_count = (boston[var] > q95).sum()
            print(f"{var}: {high_count} suburbs above 95th percentile (>{q95:.2f})")

    # (f) Charles River
    print(f"\n(f) Charles River analysis:")
    if 'chas' in boston.columns:
        charles_count = boston['chas'].sum()
        print(f"Suburbs bordering Charles River: {charles_count} out of {len(boston)}")

    # (g) Median pupil-teacher ratio
    print(f"\n(g) Median pupil-teacher ratio:")
    if 'ptratio' in boston.columns:
        median_pt = boston['ptratio'].median()
        print(f"Median pupil-teacher ratio: {median_pt}")

    # (h) Lowest median home value
    print(f"\n(h) Suburb with lowest median home value:")
    if 'medv' in boston.columns:
        min_value_idx = boston['medv'].idxmin()
        min_suburb = boston.loc[min_value_idx]
        print(f"Lowest median value: ${min_suburb['medv'] * 1000:.0f}")
        print("Characteristics of this suburb:")
        print(min_suburb)

        # Compare to overall ranges
        print(f"\nComparison to dataset ranges:")
        for col in boston.columns:
            suburb_val = min_suburb[col]
            col_min, col_max = boston[col].min(), boston[col].max()
            percentile = (boston[col] <= suburb_val).mean() * 100
            print(f"{col}: {suburb_val:.2f} (ranges {col_min:.2f}-{col_max:.2f}, {percentile:.0f}th percentile)")

    # (i) Rooms analysis
    print(f"\n(i) Room analysis:")
    if 'rm' in boston.columns:
        more_than_7 = (boston['rm'] > 7).sum()
        more_than_8 = (boston['rm'] > 8).sum()
        print(f"Suburbs with >7 rooms per dwelling: {more_than_7}")
        print(f"Suburbs with >8 rooms per dwelling: {more_than_8}")

        if more_than_8 > 0:
            print("\nCharacteristics of suburbs with >8 rooms:")
            luxury_suburbs = boston[boston['rm'] > 8]
            print(luxury_suburbs.describe())

print("\n" + "=" * 80)
print("CHAPTER 2 ANALYSIS COMPLETE!")
print("=" * 80)
print("Next steps:")
print("1. Review the conceptual questions (1-7) by hand")
print("2. Move to Chapter 3: Linear Regression")
print("3. Start with simple linear regression using Auto dataset")