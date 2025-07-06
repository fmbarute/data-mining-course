import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from ISLP import load_data

# Load Caravan data
caravan = load_data('Caravan')

# (a) Split into training (first 1000) and test sets
X = caravan.drop('Purchase', axis=1)
y = caravan['Purchase']
X_train = X.iloc[:1000]
X_test = X.iloc[1000:]
y_train = y.iloc[:1000]
y_test = y.iloc[1000:]

# Standardize features (important for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# (b) Fit boosting model
boost = GradientBoostingClassifier(n_estimators=1000,
                                 learning_rate=0.01,
                                 random_state=42)
boost.fit(X_train, y_train)

# Get feature importances
importance = pd.Series(boost.feature_importances_, index=X.columns)
print("(b) Top 10 Important Features:")
print(importance.sort_values(ascending=False).head(10))

# (c) Predict on test data with 20% threshold
boost_probs = boost.predict_proba(X_test)[:, 1]
boost_pred = np.where(boost_probs > 0.2, 'Yes', 'No')  # Convert to 'Yes'/'No'

# Confusion matrix
cm = confusion_matrix(y_test, boost_pred)
print("\n(c) Boosting Confusion Matrix:")
print(cm)

# Calculate fraction of predicted purchasers who actually purchased
precision = cm[1,1] / (cm[0,1] + cm[1,1])
print(f"\nFraction of predicted purchasers who actually purchased: {precision:.3f}")

# Compare with KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
knn_probs = knn.predict_proba(X_test_scaled)[:, 1]
knn_pred = np.where(knn_probs > 0.2, 'Yes', 'No')  # Convert to 'Yes'/'No'

cm_knn = confusion_matrix(y_test, knn_pred)
precision_knn = cm_knn[1,1] / (cm_knn[0,1] + cm_knn[1,1])
print(f"\nKNN Precision at 20% threshold: {precision_knn:.3f}")

# Compare with Logistic Regression
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_scaled, y_train)
logreg_probs = logreg.predict_proba(X_test_scaled)[:, 1]
logreg_pred = np.where(logreg_probs > 0.2, 'Yes', 'No')  # Convert to 'Yes'/'No'

cm_logreg = confusion_matrix(y_test, logreg_pred)
precision_logreg = cm_logreg[1,1] / (cm_logreg[0,1] + cm_logreg[1,1])
print(f"Logistic Regression Precision at 20% threshold: {precision_logreg:.3f}")

# Print classification reports
print("\nBoosting Classification Report:")
print(classification_report(y_test, boost_pred))

print("\nKNN Classification Report:")
print(classification_report(y_test, knn_pred))

print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, logreg_pred))