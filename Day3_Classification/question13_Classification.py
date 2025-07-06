# Complete Analysis of Weekly Stock Market Data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from statsmodels.api import add_constant
import statsmodels.api as sm

# Set random seed for reproducibility
np.random.seed(1)

# ============================================
# Data Preparation (simulating Weekly dataset)
# ============================================
print("Creating synthetic dataset similar to Weekly from ISLR2...")
n = 1089
years = np.repeat(np.arange(1990, 2011), 52)[:n]
volume = np.cumsum(np.random.normal(0.1, 0.5, n)) + 100
returns = np.random.normal(0, 2, n)
direction = np.where(returns > 0, 'Up', 'Down')

# Create lag variables
lag1 = np.roll(returns, 1)
lag2 = np.roll(returns, 2)
lag3 = np.roll(returns, 3)
lag4 = np.roll(returns, 4)
lag5 = np.roll(returns, 5)

# Fix the first values which got wrapped around
lag1[:1] = np.nan
lag2[:2] = np.nan
lag3[:3] = np.nan
lag4[:4] = np.nan
lag5[:5] = np.nan

weekly = pd.DataFrame({
    'Year': years,
    'Lag1': lag1,
    'Lag2': lag2,
    'Lag3': lag3,
    'Lag4': lag4,
    'Lag5': lag5,
    'Volume': volume,
    'Today': returns,
    'Direction': direction
})

weekly = weekly.dropna()  # Remove rows with missing lag values
weekly['Direction_bin'] = (weekly['Direction'] == 'Up').astype(int)

# Split into training (1990-2008) and test (2009-2010)
train = weekly[weekly['Year'] <= 2008]
test = weekly[weekly['Year'] >= 2009]

# ============================================
# (a) Numerical and Graphical Summaries
# ============================================
print("\n(a) Numerical and graphical summaries:")

# Numerical summaries
print("\nNumerical summaries:")
print(weekly.describe())
print("\nDirection counts:")
print(weekly['Direction'].value_counts())

# Graphical summaries
plt.figure(figsize=(15, 10))

# Time series of returns
plt.subplot(2, 2, 1)
plt.plot(weekly['Today'])
plt.title('Weekly Returns Over Time')
plt.xlabel('Week')
plt.ylabel('Return')

# Volume over time
plt.subplot(2, 2, 2)
plt.plot(weekly['Volume'])
plt.title('Trading Volume Over Time')
plt.xlabel('Week')
plt.ylabel('Volume')

# Correlation heatmap
plt.subplot(2, 2, 3)
corr = weekly[['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume', 'Today']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')

# Direction distribution
plt.subplot(2, 2, 4)
weekly['Direction'].value_counts().plot(kind='bar')
plt.title('Direction Distribution')

plt.tight_layout()
plt.show()

# ============================================
# (b) Logistic Regression with All Predictors
# ============================================
print("\n(b) Logistic regression with all predictors:")
X = weekly[['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume']]
X = add_constant(X)  # Add intercept
y = weekly['Direction_bin']

logit_model = sm.Logit(y, X)
result = logit_model.fit()
print(result.summary())

# ============================================
# (c) Confusion Matrix and Accuracy
# ============================================
print("\n(c) Confusion matrix and accuracy:")
predictions = result.predict(X) > 0.5

cm = confusion_matrix(y, predictions)
print("Confusion Matrix:")
print(cm)

accuracy = accuracy_score(y, predictions)
print(f"\nOverall Accuracy: {accuracy:.4f}")

# ============================================
# (d) Logistic Regression with Training/Test Split (Lag2 only)
# ============================================
print("\n(d) Logistic regression with Lag2 only:")
X_train = train[['Lag2']]
X_train = add_constant(X_train)
y_train = train['Direction_bin']

X_test = test[['Lag2']]
X_test = add_constant(X_test)
y_test = test['Direction_bin']

logit_model2 = sm.Logit(y_train, X_train)
result2 = logit_model2.fit()

test_predictions = result2.predict(X_test) > 0.5

cm = confusion_matrix(y_test, test_predictions)
print("Confusion Matrix:")
print(cm)

accuracy = accuracy_score(y_test, test_predictions)
print(f"\nTest Accuracy: {accuracy:.4f}")

# ============================================
# (e) LDA
# ============================================
print("\n(e) Linear Discriminant Analysis:")
lda = LinearDiscriminantAnalysis()
lda.fit(train[['Lag2']], train['Direction'])

lda_predictions = lda.predict(test[['Lag2']])

cm = confusion_matrix(test['Direction'], lda_predictions)
print("LDA Confusion Matrix:")
print(cm)

accuracy = accuracy_score(test['Direction'], lda_predictions)
print(f"\nLDA Test Accuracy: {accuracy:.4f}")

# ============================================
# (f) QDA
# ============================================
print("\n(f) Quadratic Discriminant Analysis:")
qda = QuadraticDiscriminantAnalysis()
qda.fit(train[['Lag2']], train['Direction'])

qda_predictions = qda.predict(test[['Lag2']])

cm = confusion_matrix(test['Direction'], qda_predictions)
print("QDA Confusion Matrix:")
print(cm)

accuracy = accuracy_score(test['Direction'], qda_predictions)
print(f"\nQDA Test Accuracy: {accuracy:.4f}")

# ============================================
# (g) KNN with K=1
# ============================================
print("\n(g) K-Nearest Neighbors (K=1):")
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(train[['Lag2']], train['Direction'])

knn_predictions = knn.predict(test[['Lag2']])

cm = confusion_matrix(test['Direction'], knn_predictions)
print("KNN (K=1) Confusion Matrix:")
print(cm)

accuracy = accuracy_score(test['Direction'], knn_predictions)
print(f"\nKNN (K=1) Test Accuracy: {accuracy:.4f}")

# ============================================
# (h) Naive Bayes
# ============================================
print("\n(h) Naive Bayes:")
nb = GaussianNB()
nb.fit(train[['Lag2']], train['Direction'])

nb_predictions = nb.predict(test[['Lag2']])

cm = confusion_matrix(test['Direction'], nb_predictions)
print("Naive Bayes Confusion Matrix:")
print(cm)

accuracy = accuracy_score(test['Direction'], nb_predictions)
print(f"\nNaive Bayes Test Accuracy: {accuracy:.4f}")

# ============================================
# (i) Comparison of Methods
# ============================================
print("\n(i) Comparison of methods:")
methods = {
    'Logistic Regression': accuracy_score(y_test, test_predictions),
    'LDA': accuracy_score(test['Direction'], lda_predictions),
    'QDA': accuracy_score(test['Direction'], qda_predictions),
    'KNN (K=1)': accuracy_score(test['Direction'], knn_predictions),
    'Naive Bayes': accuracy_score(test['Direction'], nb_predictions)
}

print("\nTest Accuracies:")
for method, acc in methods.items():
    print(f"{method}: {acc:.4f}")

best_method = max(methods, key=methods.get)
print(f"\nBest method on test data: {best_method} with accuracy {methods[best_method]:.4f}")

# ============================================
# (j) Experimentation with Different Predictors
# ============================================
print("\n(j) Experimenting with different predictors...")

# Try different combinations
print("\nTrying Lag2 + Volume:")
X_train = train[['Lag2', 'Volume']]
X_train = add_constant(X_train)
y_train = train['Direction_bin']

X_test = test[['Lag2', 'Volume']]
X_test = add_constant(X_test)
y_test = test['Direction_bin']

logit_model_j1 = sm.Logit(y_train, X_train)
result_j1 = logit_model_j1.fit()
test_predictions_j1 = result_j1.predict(X_test) > 0.5
accuracy = accuracy_score(y_test, test_predictions_j1)
print(f"LogReg (Lag2+Volume) Test Accuracy: {accuracy:.4f}")

# Try KNN with different K values
print("\nKNN with different K values:")
for k in [1, 3, 5, 10, 50]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train[['Lag2', 'Volume']], train['Direction'])
    knn_predictions = knn.predict(test[['Lag2', 'Volume']])
    accuracy = accuracy_score(test['Direction'], knn_predictions)
    print(f"KNN with K={k} Accuracy: {accuracy:.4f}")

# Try interaction term
print("\nTrying Lag2 * Volume interaction:")
train['Lag2_Volume'] = train['Lag2'] * train['Volume']
test['Lag2_Volume'] = test['Lag2'] * test['Volume']

X_train = train[['Lag2', 'Volume', 'Lag2_Volume']]
X_train = add_constant(X_train)
y_train = train['Direction_bin']

X_test = test[['Lag2', 'Volume', 'Lag2_Volume']]
X_test = add_constant(X_test)
y_test = test['Direction_bin']

logit_model_j2 = sm.Logit(y_train, X_train)
result_j2 = logit_model_j2.fit()
test_predictions_j2 = result_j2.predict(X_test) > 0.5
accuracy = accuracy_score(y_test, test_predictions_j2)
print(f"LogReg (Lag2*Volume interaction) Test Accuracy: {accuracy:.4f}")

print("\nAnalysis complete!")