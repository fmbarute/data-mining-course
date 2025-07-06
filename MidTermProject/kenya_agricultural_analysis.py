import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, VotingRegressor, VotingClassifier
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, RFE
from sklearn.metrics import (classification_report, mean_absolute_error,
                             mean_squared_error, r2_score, silhouette_score,
                             calinski_harabasz_score, davies_bouldin_score,
                             accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("Kenya Agricultural Survey Analysis - MSDA9223")
print("=" * 55)
print("Dataset: Agricultural Survey of African Farm Households")
print("Focus Country: Kenya (EAC Member)")
print("=" * 55)


# =============================================================================
# 1. DATA LOADING AND KENYA-SPECIFIC EXPLORATION - FIXED
# =============================================================================

def load_kenya_agricultural_data(file_path):

    print("Loading Agricultural Survey Dataset...")

    # Load the complete dataset
    df = pd.read_csv(file_path)

    print(f"Complete Dataset Shape: {df.shape}")

    # Display first few columns to understand structure
    print(f"\nFirst 10 columns: {list(df.columns[:10])}")

    country_columns = [col for col in df.columns if 'country' in col.lower() or 'COUNTRY' in col]

    if country_columns:
        country_col = country_columns[0]
        print(f"Country column found: {country_col}")
        print(f"Available countries: {df[country_col].unique()}")

        # Filter for Kenya
        kenya_indicators = ['Kenya', 'KENYA', 'KEN', 'kenya']
        kenya_df = df[df[country_col].isin(kenya_indicators)]

        if len(kenya_df) == 0:
            # Try partial matching
            kenya_df = df[df[country_col].str.contains('Kenya|KENYA|ken', case=False, na=False)]

        print(f"Kenya Dataset Shape: {kenya_df.shape}")

    else:
        print("Country column not found. Checking for other identifiers...")

        possible_country_cols = [col for col in df.columns if
                                 any(word in col.lower() for word in ['nation', 'region', 'location', 'site'])]

        if possible_country_cols:
            print(f"Found possible location columns: {possible_country_cols}")
            for col in possible_country_cols:
                unique_vals = df[col].unique()
                print(f"{col}: {unique_vals}")

                # Check if Kenya is mentioned
                kenya_mask = df[col].astype(str).str.contains('Kenya|kenya|KEN', case=False, na=False)
                if kenya_mask.sum() > 0:
                    kenya_df = df[kenya_mask]
                    print(f"Found {len(kenya_df)} Kenya records using column: {col}")
                    break
            else:
                print("No Kenya data found. Using complete dataset...")
                kenya_df = df
        else:
            print("No country identifiers found. Using complete dataset...")
            kenya_df = df

    print("\nKenya Dataset Overview:")
    print(f"Number of households: {len(kenya_df)}")
    print(f"Number of variables: {len(kenya_df.columns)}")

    return kenya_df


def explore_agricultural_variables(df):

    print("Exploring Key Agricultural Variables...")

    # Key agricultural indicators to look for
    agricultural_keywords = [
        'yield', 'production', 'crop', 'harvest', 'income', 'area', 'land',
        'fertilizer', 'seed', 'irrigation', 'livestock', 'cattle', 'goat',
        'rainfall', 'weather', 'climate', 'drought', 'food', 'maize', 'rice'
    ]

    # Find relevant columns
    relevant_columns = []
    for keyword in agricultural_keywords:
        matching_cols = [col for col in df.columns if keyword.lower() in col.lower()]
        relevant_columns.extend(matching_cols)

    # Remove duplicates
    relevant_columns = list(set(relevant_columns))

    print(f"Found {len(relevant_columns)} potentially relevant agricultural columns:")
    for i, col in enumerate(relevant_columns[:20]):  # Show first 20
        print(f"{i + 1:2d}. {col}")

    if len(relevant_columns) > 20:
        print(f"... and {len(relevant_columns) - 20} more columns")

    # Analyze data types and missing values for key columns
    if relevant_columns:
        print(f"\nData Quality Analysis for Key Variables:")
        print("-" * 50)

        sample_cols = relevant_columns[:10]  # Analyze first 10 relevant columns
        for col in sample_cols:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            dtype = df[col].dtype
            unique_vals = df[col].nunique()

            print(f"{col[:40]:<40} | {str(dtype):<10} | Missing: {missing_pct:5.1f}% | Unique: {unique_vals:4d}")


# =============================================================================
# 2. KENYA AGRICULTURAL DATA CLEANING AND PREPROCESSING
# =============================================================================

def clean_kenya_agricultural_data(df):

    print("Cleaning Kenya Agricultural Data...")

    df_clean = df.copy()

    print(f"Initial shape: {df_clean.shape}")

    # 1. Remove columns with >80% missing values
    missing_threshold = 0.8
    missing_percentages = df_clean.isnull().sum() / len(df_clean)
    columns_to_keep = missing_percentages[missing_percentages <= missing_threshold].index
    df_clean = df_clean[columns_to_keep]

    print(f"After removing high-missing columns: {df_clean.shape}")

    # 2. Remove rows with >50% missing values
    row_missing_threshold = 0.5
    df_clean = df_clean.dropna(thresh=int(row_missing_threshold * len(df_clean.columns)))

    print(f"After removing incomplete rows: {df_clean.shape}")

    # 3. Handle specific agricultural data issues
    for column in df_clean.columns:
        if df_clean[column].dtype in ['object']:
            # Fill categorical with mode or 'Unknown'
            mode_val = df_clean[column].mode()
            fill_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
            df_clean[column].fillna(fill_val, inplace=True)
        else:
            # Fill numerical with median (better for agricultural data with outliers)
            df_clean[column].fillna(df_clean[column].median(), inplace=True)

    # 4. Remove obvious duplicates
    df_clean.drop_duplicates(inplace=True)

    print(f"Final cleaned shape: {df_clean.shape}")

    return df_clean


def create_kenya_agricultural_features(df):

    print("Creating Kenya Agricultural Features...")

    df_features = df.copy()

    yield_cols = [col for col in df_features.columns if 'yield' in col.lower()]
    production_cols = [col for col in df_features.columns if 'production' in col.lower() or 'harvest' in col.lower()]
    area_cols = [col for col in df_features.columns if 'area' in col.lower() or 'land' in col.lower()]

    print(
        f"Found {len(yield_cols)} yield columns, {len(production_cols)} production columns, {len(area_cols)} area columns")

    # Create composite productivity index
    if yield_cols:
        # Standardize and combine yield columns
        yield_data = df_features[yield_cols].select_dtypes(include=[np.number])
        if not yield_data.empty:
            yield_standardized = (yield_data - yield_data.mean()) / yield_data.std()
            df_features['productivity_index'] = yield_standardized.mean(axis=1)
            print("✅ Created productivity_index")

    # 2. Income and Economic Indicators
    income_cols = [col for col in df_features.columns if 'income' in col.lower() or 'profit' in col.lower()]
    print(f"Found {len(income_cols)} income-related columns")

    if income_cols:
        income_data = df_features[income_cols].select_dtypes(include=[np.number])
        if not income_data.empty:
            df_features['total_agricultural_income'] = income_data.sum(axis=1)
            df_features['log_income'] = np.log1p(df_features['total_agricultural_income'])
            print("✅ Created total_agricultural_income and log_income")

    # 3. Technology Adoption Index
    tech_keywords = ['fertilizer', 'irrigation', 'mechaniz', 'improved', 'hybrid']
    tech_cols = []
    for keyword in tech_keywords:
        tech_cols.extend([col for col in df_features.columns if keyword.lower() in col.lower()])

    print(f"Found {len(tech_cols)} technology-related columns")

    if tech_cols:
        tech_data = df_features[tech_cols].select_dtypes(include=[np.number])
        if not tech_data.empty:
            # Binary conversion (assuming 0/1 or Yes/No type data)
            tech_binary = (tech_data > 0).astype(int)
            df_features['technology_adoption_score'] = tech_binary.sum(axis=1)
            print("✅ Created technology_adoption_score")

    # 4. Livestock Diversification
    livestock_cols = [col for col in df_features.columns
                      if any(animal in col.lower() for animal in ['cattle', 'goat', 'sheep', 'chicken', 'livestock'])]

    print(f"Found {len(livestock_cols)} livestock-related columns")

    if livestock_cols:
        livestock_data = df_features[livestock_cols].select_dtypes(include=[np.number])
        if not livestock_data.empty:
            df_features['livestock_diversity'] = (livestock_data > 0).sum(axis=1)
            df_features['total_livestock_units'] = livestock_data.sum(axis=1)
            print("✅ Created livestock_diversity and total_livestock_units")

    # 5. Climate Resilience Indicators
    climate_cols = [col for col in df_features.columns
                    if any(climate in col.lower() for climate in ['drought', 'rainfall', 'weather', 'climate'])]

    print(f"Found {len(climate_cols)} climate-related columns")

    # 6. Food Security Index
    food_cols = [col for col in df_features.columns if 'food' in col.lower() or 'nutrition' in col.lower()]
    print(f"Found {len(food_cols)} food security columns")

    print(f"Features created. New shape: {df_features.shape}")

    # Create target variables for modeling

    # Target 1: Agricultural Income (Regression)
    if 'total_agricultural_income' in df_features.columns:
        df_features['target_income'] = df_features['total_agricultural_income']
        print("✅ Created target_income from total_agricultural_income")
    elif income_cols:
        # Use first available income column
        first_income_col = income_cols[0]
        if df_features[first_income_col].dtype in [np.number]:
            df_features['target_income'] = df_features[first_income_col]
            print(f"✅ Created target_income from {first_income_col}")

    # Target 2: Farm Productivity Category (Classification)
    if 'productivity_index' in df_features.columns:
        # Create productivity categories based on quartiles
        productivity_quartiles = df_features['productivity_index'].quantile([0.25, 0.5, 0.75])
        df_features['productivity_category'] = pd.cut(df_features['productivity_index'],
                                                      bins=[-np.inf, productivity_quartiles[0.25],
                                                            productivity_quartiles[0.75], np.inf],
                                                      labels=['Low', 'Medium', 'High'])
        print("✅ Created productivity_category from productivity_index")

    # If no clear productivity measure, create based on income
    elif 'target_income' in df_features.columns:
        income_quartiles = df_features['target_income'].quantile([0.33, 0.67])
        df_features['productivity_category'] = pd.cut(df_features['target_income'],
                                                      bins=[-np.inf, income_quartiles[0.33],
                                                            income_quartiles[0.67], np.inf],
                                                      labels=['Low', 'Medium', 'High'])
        print("✅ Created productivity_category from target_income")

    return df_features


# [Keep all other functions the same - just update the main function]

# =============================================================================
# 6. MAIN EXECUTION FOR KENYA AGRICULTURAL ANALYSIS - FIXED
# =============================================================================

def main_kenya_agricultural_analysis():

    print("Kenya Agricultural Analysis - Complete Pipeline")
    print("=" * 60)

    dataset_path = '../Data/Agricultural_Survey_of_African_Farm_Households.csv'

    try:
        # Step 1: Load Kenya data
        print("\n1. LOADING KENYA AGRICULTURAL DATA")
        print("-" * 40)
        kenya_df = load_kenya_agricultural_data(dataset_path)
        explore_agricultural_variables(kenya_df)

        # Step 2: Clean data
        print("\n2. CLEANING AGRICULTURAL DATA")
        print("-" * 35)
        kenya_clean = clean_kenya_agricultural_data(kenya_df)

        # Step 3: Feature engineering
        print("\n3. AGRICULTURAL FEATURE ENGINEERING")
        print("-" * 40)
        kenya_features = create_kenya_agricultural_features(kenya_clean)

        # Check if we have the required target variables
        has_regression_target = 'target_income' in kenya_features.columns
        has_classification_target = 'productivity_category' in kenya_features.columns

        print(f"\nTarget Variables Status:")
        print(f"Regression target (income): {'✅' if has_regression_target else '❌'}")
        print(f"Classification target (productivity): {'✅' if has_classification_target else '❌'}")

        # Step 4: EDA
        print("\n4. KENYA AGRICULTURAL EDA")
        print("-" * 30)
        perform_kenya_agricultural_eda(kenya_features)

        # Step 5: Prepare modeling data
        print("\n5. PREPARING MODELING DATA")
        print("-" * 32)
        X, y_regression, y_classification, feature_names = prepare_kenya_modeling_data(kenya_features)

        # Step 6: Regression modeling
        if y_regression is not None and len(y_regression.dropna()) > 50:
            print("\n6. AGRICULTURAL INCOME PREDICTION")
            print("-" * 40)
            reg_models, reg_results, reg_scaler, reg_splits = build_agricultural_regression_models(X, y_regression,
                                                                                                   feature_names)
        else:
            print("\n6. REGRESSION MODELING SKIPPED")
            print("Reason: Insufficient regression target data")
            reg_results = None

        # Step 7: Classification modeling
        if y_classification is not None and len(y_classification.dropna()) > 50:
            print("\n7. PRODUCTIVITY CLASSIFICATION")
            print("-" * 35)
            class_models, class_results, class_scaler, class_splits = build_agricultural_classification_models(X,
                                                                                                               y_classification,
                                                                                                               feature_names)
        else:
            print("\n7. CLASSIFICATION MODELING SKIPPED")
            print("Reason: Insufficient classification target data")
            class_results = None

        # Step 8: Clustering analysis (this should always work)
        print("\n8. HOUSEHOLD CLUSTERING ANALYSIS")
        print("-" * 38)
        clustering_results, cluster_scaler = perform_agricultural_clustering(X, feature_names)

        print("\n" + "=" * 60)
        print("KENYA AGRICULTURAL ANALYSIS COMPLETE!")
        print("=" * 60)

        return {
            'data': kenya_features,
            'regression_results': reg_results,
            'classification_results': class_results,
            'clustering_results': clustering_results
        }

    except FileNotFoundError:
        print(f"File not found: {dataset_path}")
        print("Available files in directory:")
        import os
        files = [f for f in os.listdir('.') if f.endswith('.csv')]
        for f in files:
            print(f"  - {f}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        print("Please check the error details above")


def perform_kenya_agricultural_eda(df):

    print("Performing Kenya Agricultural EDA...")

    plt.style.use('default')
    sns.set_palette("viridis")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    if 'target_income' in df.columns:
        income_data = df['target_income'].dropna()
        if len(income_data) > 0:
            axes[0, 0].hist(income_data, bins=50, alpha=0.7, color='green')
            axes[0, 0].set_title('Agricultural Income Distribution')
            axes[0, 0].set_xlabel('Income')
            axes[0, 0].set_ylabel('Frequency')

            # Log-transformed income
            log_income = np.log1p(income_data)
            axes[0, 1].hist(log_income, bins=50, alpha=0.7, color='darkgreen')
            axes[0, 1].set_title('Log-Transformed Income Distribution')
            axes[0, 1].set_xlabel('Log(Income + 1)')
            axes[0, 1].set_ylabel('Frequency')
        else:
            axes[0, 0].text(0.5, 0.5, 'No Income Data', ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 1].text(0.5, 0.5, 'No Income Data', ha='center', va='center', transform=axes[0, 1].transAxes)
    else:
        axes[0, 0].text(0.5, 0.5, 'No Income Data', ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 1].text(0.5, 0.5, 'No Income Data', ha='center', va='center', transform=axes[0, 1].transAxes)

    # Productivity categories
    if 'productivity_category' in df.columns:
        productivity_counts = df['productivity_category'].value_counts()
        if len(productivity_counts) > 0:
            axes[0, 2].pie(productivity_counts.values, labels=productivity_counts.index, autopct='%1.1f%%')
            axes[0, 2].set_title('Farm Productivity Categories')
        else:
            axes[0, 2].text(0.5, 0.5, 'No Productivity Data', ha='center', va='center', transform=axes[0, 2].transAxes)
    else:
        axes[0, 2].text(0.5, 0.5, 'No Productivity Data', ha='center', va='center', transform=axes[0, 2].transAxes)

    # Technology adoption
    if 'technology_adoption_score' in df.columns:
        tech_data = df['technology_adoption_score'].dropna()
        if len(tech_data) > 0:
            axes[1, 0].hist(tech_data, bins=20, alpha=0.7, color='blue')
            axes[1, 0].set_title('Technology Adoption Score Distribution')
            axes[1, 0].set_xlabel('Technology Adoption Score')
            axes[1, 0].set_ylabel('Frequency')
        else:
            axes[1, 0].text(0.5, 0.5, 'No Technology Data', ha='center', va='center', transform=axes[1, 0].transAxes)
    else:
        axes[1, 0].text(0.5, 0.5, 'No Technology Data', ha='center', va='center', transform=axes[1, 0].transAxes)

    # Livestock diversity
    if 'livestock_diversity' in df.columns:
        livestock_data = df['livestock_diversity'].dropna()
        if len(livestock_data) > 0:
            axes[1, 1].hist(livestock_data, bins=15, alpha=0.7, color='orange')
            axes[1, 1].set_title('Livestock Diversity Distribution')
            axes[1, 1].set_xlabel('Number of Livestock Types')
            axes[1, 1].set_ylabel('Frequency')
        else:
            axes[1, 1].text(0.5, 0.5, 'No Livestock Data', ha='center', va='center', transform=axes[1, 1].transAxes)
    else:
        axes[1, 1].text(0.5, 0.5, 'No Livestock Data', ha='center', va='center', transform=axes[1, 1].transAxes)

    # Productivity vs Technology
    if 'productivity_index' in df.columns and 'technology_adoption_score' in df.columns:
        scatter_data = df[['productivity_index', 'technology_adoption_score']].dropna()
        if len(scatter_data) > 0:
            axes[1, 2].scatter(scatter_data['technology_adoption_score'],
                               scatter_data['productivity_index'], alpha=0.6, color='red')
            axes[1, 2].set_title('Productivity vs Technology Adoption')
            axes[1, 2].set_xlabel('Technology Adoption Score')
            axes[1, 2].set_ylabel('Productivity Index')
        else:
            axes[1, 2].text(0.5, 0.5, 'No Scatter Data', ha='center', va='center', transform=axes[1, 2].transAxes)
    else:
        axes[1, 2].text(0.5, 0.5, 'No Scatter Data', ha='center', va='center', transform=axes[1, 2].transAxes)

    plt.tight_layout()
    plt.suptitle('Kenya Agricultural Survey - Key Indicators', y=1.02, fontsize=16)
    plt.show()

    # 3. Summary Statistics
    print("\nKenya Agricultural Survey - Key Insights:")
    print("-" * 50)

    if 'target_income' in df.columns and df['target_income'].notna().sum() > 0:
        income_stats = df['target_income'].describe()
        print(f"Agricultural Income Statistics:")
        print(f"  Mean: {income_stats['mean']:.2f}")
        print(f"  Median: {income_stats['50%']:.2f}")
        print(f"  Std Dev: {income_stats['std']:.2f}")
    else:
        print("No income data available for analysis")

    if 'productivity_category' in df.columns:
        prod_counts = df['productivity_category'].value_counts()
        if len(prod_counts) > 0:
            print(f"\nProductivity Distribution:")
            for category, count in prod_counts.items():
                print(f"  {category}: {count} households ({count / len(df) * 100:.1f}%)")

    if 'technology_adoption_score' in df.columns and df['technology_adoption_score'].notna().sum() > 0:
        tech_mean = df['technology_adoption_score'].mean()
        print(f"\nAverage Technology Adoption Score: {tech_mean:.2f}")


def prepare_kenya_modeling_data(df):

    print("Preparing Data for Machine Learning...")


    exclude_columns = ['target_income', 'productivity_category', 'productivity_index', 'total_agricultural_income',
                       'log_income']

    # Get feature columns
    feature_columns = [col for col in df.columns if col not in exclude_columns]

    # Handle categorical variables
    df_modeling = df.copy()

    # Encode categorical variables
    categorical_columns = df_modeling[feature_columns].select_dtypes(include=['object']).columns

    for col in categorical_columns:
        if df_modeling[col].nunique() < 50:  # Only encode if reasonable number of categories
            try:
                # Create dummy variables
                dummies = pd.get_dummies(df_modeling[col], prefix=col, drop_first=True)
                df_modeling = pd.concat([df_modeling, dummies], axis=1)
                feature_columns.remove(col)
                feature_columns.extend(dummies.columns)
            except Exception as e:
                print(f"Warning: Could not encode column {col}: {e}")
                feature_columns.remove(col)

    # Remove non-numeric columns that couldn't be encoded
    feature_columns = [col for col in feature_columns if col in df_modeling.columns]
    numeric_features = df_modeling[feature_columns].select_dtypes(include=[np.number]).columns
    feature_columns = list(numeric_features)

    # Prepare feature matrix
    X = df_modeling[feature_columns].fillna(0)  # Fill any remaining NaN values

    # Remove constant columns
    constant_columns = [col for col in X.columns if X[col].nunique() <= 1]
    if constant_columns:
        print(f"Removing {len(constant_columns)} constant columns")
        X = X.drop(columns=constant_columns)
        feature_columns = [col for col in feature_columns if col not in constant_columns]

    # Prepare target variables
    y_regression = None
    y_classification = None

    if 'target_income' in df_modeling.columns:
        y_regression = df_modeling['target_income'].fillna(df_modeling['target_income'].median())
        # Remove rows where target is still NaN or zero
        valid_regression_mask = (y_regression.notna()) & (y_regression > 0)
        if valid_regression_mask.sum() > 0:
            print(f"Valid regression samples: {valid_regression_mask.sum()}")
        else:
            y_regression = None
            print("No valid regression target data")

    if 'productivity_category' in df_modeling.columns:
        y_classification = df_modeling['productivity_category'].fillna('Medium')
        # Remove rows where classification target is NaN
        valid_classification_mask = y_classification.notna()
        if valid_classification_mask.sum() > 0:
            print(f"Valid classification samples: {valid_classification_mask.sum()}")
        else:
            y_classification = None
            print("No valid classification target data")

    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of features: {len(feature_columns)}")

    if y_regression is not None:
        print(f"Regression target (income) - Non-null values: {y_regression.notna().sum()}")

    if y_classification is not None:
        print(f"Classification target (productivity) - Distribution:")
        print(y_classification.value_counts())

    return X, y_regression, y_classification, feature_columns


def build_agricultural_regression_models(X, y, feature_names):
    """Build regression models for Kenya agricultural income prediction"""

    print("Building Agricultural Income Prediction Models...")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {}
    results = {}

    # 1. Linear Regression
    print("Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    models['Linear Regression'] = lr

    # 2. Random Forest
    print("Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf

    # 3. Support Vector Regression
    print("Training SVR...")
    svr = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
    svr.fit(X_train_scaled, y_train)
    models['SVR'] = svr

    # 4. Deep Neural Network for Agricultural Data
    print("Training Agricultural Deep Neural Network...")
    dnn = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='linear')
    ])

    dnn.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    history = dnn.fit(X_train_scaled, y_train,
                      epochs=100,
                      batch_size=32,
                      validation_split=0.2,
                      callbacks=[early_stopping],
                      verbose=0)

    models['Deep Neural Network'] = dnn

    # Evaluate models
    for name, model in models.items():
        try:
            if name == 'Deep Neural Network':
                y_pred = model.predict(X_test_scaled).flatten()
            elif name in ['Linear Regression', 'SVR']:
                y_pred = model.predict(X_test_scaled)
            else:
                y_pred = model.predict(X_test)

            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            results[name] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'predictions': y_pred
            }

            print(f"{name}: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}")
        except Exception as e:
            print(f"Error training {name}: {e}")

    # Feature importance analysis for Random Forest
    if 'Random Forest' in models:
        try:
            rf_model = models['Random Forest']
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)

            print("\nTop 10 Most Important Features for Agricultural Income:")
            print(feature_importance.head(10))
        except Exception as e:
            print(f"Error calculating feature importance: {e}")

    return models, results, scaler, (X_train, X_test, y_train, y_test)


def build_agricultural_classification_models(X, y, feature_names):
    """Build classification models for Kenya agricultural productivity classification"""

    print("Building Agricultural Productivity Classification Models...")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {}
    results = {}

    # 1. Logistic Regression
    print("Training Logistic Regression...")
    lr = LogisticRegression(random_state=42, max_iter=1000, multi_class='ovr')
    lr.fit(X_train_scaled, y_train)
    models['Logistic Regression'] = lr

    # 2. Random Forest Classifier
    print("Training Random Forest Classifier...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf

    # 3. Support Vector Machine
    print("Training SVM...")
    svm = SVC(kernel='rbf', probability=True, random_state=42, C=10)
    svm.fit(X_train_scaled, y_train)
    models['SVM'] = svm

    # 4. Deep Neural Network for Classification
    print("Training Classification Deep Neural Network...")

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    n_classes = len(label_encoder.classes_)

    dnn = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(n_classes, activation='softmax')
    ])

    dnn.compile(optimizer=Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    history = dnn.fit(X_train_scaled, y_train_encoded,
                      epochs=100,
                      batch_size=32,
                      validation_split=0.2,
                      callbacks=[early_stopping],
                      verbose=0)

    models['Deep Neural Network'] = dnn

    # Evaluate models
    for name, model in models.items():
        try:
            if name == 'Deep Neural Network':
                y_pred_proba = model.predict(X_test_scaled)
                y_pred = label_encoder.inverse_transform(np.argmax(y_pred_proba, axis=1))
            elif name in ['Logistic Regression', 'SVM']:
                y_pred = model.predict(X_test_scaled)
            else:
                y_pred = model.predict(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            results[name] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'predictions': y_pred
            }

            print(f"{name}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        except Exception as e:
            print(f"Error training {name}: {e}")

    return models, results, scaler, (X_train, X_test, y_train, y_test)


def perform_agricultural_clustering(X, feature_names):
    """Perform clustering analysis for Kenya agricultural households"""

    print("Performing Agricultural Household Clustering...")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use smaller range for large datasets
    K_range = range(2, 8)

    # Determine optimal number of clusters
    inertias = []
    silhouette_scores = []

    for k in K_range:
        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans_temp.fit_predict(X_scaled)
        inertias.append(kmeans_temp.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))

    # Plot elbow curve
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(K_range, inertias, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method - Agricultural Clusters')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(K_range, silhouette_scores, 'ro-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score - Agricultural Clusters')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Use optimal number of clusters (5 for agricultural typology)
    n_clusters = 5

    clustering_results = {}

    # 1. K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)

    # 2. Hierarchical Clustering
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    hierarchical_labels = hierarchical.fit_predict(X_scaled)

    # 3. DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_scaled)

    # Evaluate clustering
    methods = {
        'K-Means': kmeans_labels,
        'Hierarchical': hierarchical_labels,
        'DBSCAN': dbscan_labels
    }

    for method, labels in methods.items():
        if len(set(labels)) > 1:
            try:
                silhouette_avg = silhouette_score(X_scaled, labels)
                calinski_harabasz = calinski_harabasz_score(X_scaled, labels)
                davies_bouldin = davies_bouldin_score(X_scaled, labels)
                n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)

                clustering_results[method] = {
                    'labels': labels,
                    'silhouette_score': silhouette_avg,
                    'calinski_harabasz_score': calinski_harabasz,
                    'davies_bouldin_score': davies_bouldin,
                    'n_clusters': n_clusters_found
                }

                print(f"{method}: Silhouette={silhouette_avg:.4f}, "
                      f"Calinski-Harabasz={calinski_harabasz:.2f}, "
                      f"Davies-Bouldin={davies_bouldin:.4f}, "
                      f"Clusters={n_clusters_found}")
            except Exception as e:
                print(f"Error evaluating {method}: {e}")

    # Analyze K-Means clusters
    if 'K-Means' in clustering_results:
        try:
            cluster_df = pd.DataFrame(X)
            cluster_df.columns = feature_names
            cluster_df['cluster'] = kmeans_labels

            print("\nK-Means Cluster Analysis:")
            print("Cluster sizes:", pd.Series(kmeans_labels).value_counts().sort_index())

            # Show cluster centroids for key features
            key_features = feature_names[:5]  # Top 5 features
            cluster_means = cluster_df.groupby('cluster')[key_features].mean()
            print("\nCluster Centroids (Key Features):")
            print(cluster_means)
        except Exception as e:
            print(f"Error in cluster analysis: {e}")

    return clustering_results, scaler


if __name__ == "__main__":
    results = main_kenya_agricultural_analysis()