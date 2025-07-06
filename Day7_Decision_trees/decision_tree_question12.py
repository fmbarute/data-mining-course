import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (GradientBoostingClassifier,
                              BaggingClassifier,
                              RandomForestClassifier)
from sklearn.tree import DecisionTreeClassifier
from ISLP import load_data

# Attempt to import BART (if available)
try:
    from bartpy.sklearnmodel import SklearnModel

    BART_AVAILABLE = True
except ImportError:
    print("BART not available (requires bartpy). Skipping BART implementation.")
    BART_AVAILABLE = False


# Load and prepare data
def load_and_prepare_data():
    caravan = load_data('Caravan')
    X = caravan.drop('Purchase', axis=1)
    y = caravan['Purchase'].map({'No': 0, 'Yes': 1})  # Convert to binary (0/1)

    # Train-test split (first 1000 = train, rest = test)
    X_train, X_test = X.iloc[:1000], X.iloc[1000:]
    y_train, y_test = y.iloc[:1000], y.iloc[1000:]

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled


# Logistic Regression
def run_logistic_regression(X_train_scaled, X_test_scaled, y_train, y_test):
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train_scaled, y_train)
    logreg_pred = logreg.predict(X_test_scaled)

    print("\nLogistic Regression Results:")
    print("Accuracy:", accuracy_score(y_test, logreg_pred))
    print(classification_report(y_test, logreg_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, logreg_pred))

    return logreg_pred


# Gradient Boosting
def run_boosting(X_train, X_test, y_train, y_test):
    boost = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.01, random_state=42)
    boost.fit(X_train, y_train)
    boost_pred = boost.predict(X_test)

    print("\nBoosting Results:")
    print("Accuracy:", accuracy_score(y_test, boost_pred))
    print(classification_report(y_test, boost_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, boost_pred))

    return boost_pred


# Bagging
def run_bagging(X_train, X_test, y_train, y_test):
    bagging = BaggingClassifier(
        estimator=DecisionTreeClassifier(),  # Changed from base_estimator to estimator
        n_estimators=500,
        random_state=42
    )
    bagging.fit(X_train, y_train)
    bagging_pred = bagging.predict(X_test)

    print("\nBagging Results:")
    print("Accuracy:", accuracy_score(y_test, bagging_pred))
    print(classification_report(y_test, bagging_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, bagging_pred))

    return bagging_pred


# Random Forest
def run_random_forest(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=500, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    print("\nRandom Forest Results:")
    print("Accuracy:", accuracy_score(y_test, rf_pred))
    print(classification_report(y_test, rf_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, rf_pred))

    return rf_pred


# BART (if available)
def run_bart(X_train, X_test, y_train, y_test):
    if not BART_AVAILABLE:
        print("\nSkipping BART (package not available)")
        return None

    bart = SklearnModel(n_trees=50, n_chains=4, n_samples=200)
    bart.fit(X_train, y_train)
    bart_pred = bart.predict(X_test).round().astype(int)

    print("\nBART Results:")
    print("Accuracy:", accuracy_score(y_test, bart_pred))
    print(classification_report(y_test, bart_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, bart_pred))

    return bart_pred


# Main execution
if __name__ == "__main__":
    # Load and prepare data
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = load_and_prepare_data()

    # Run models
    print("=== Model Comparison on Caravan Dataset ===")
    lr_pred = run_logistic_regression(X_train_scaled, X_test_scaled, y_train, y_test)
    boost_pred = run_boosting(X_train, X_test, y_train, y_test)
    bagging_pred = run_bagging(X_train, X_test, y_train, y_test)
    rf_pred = run_random_forest(X_train, X_test, y_train, y_test)
    bart_pred = run_bart(X_train, X_test, y_train, y_test)

    # Summary of results
    print("\n=== Summary of Results ===")
    results = {
        'Logistic Regression': accuracy_score(y_test, lr_pred),
        'Boosting': accuracy_score(y_test, boost_pred),
        'Bagging': accuracy_score(y_test, bagging_pred),
        'Random Forest': accuracy_score(y_test, rf_pred),
    }

    if BART_AVAILABLE and bart_pred is not None:
        results['BART'] = accuracy_score(y_test, bart_pred)

    results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Accuracy'])
    print(results_df.sort_values('Accuracy', ascending=False))