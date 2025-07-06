import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# Load the dataset
url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
titanic = pd.read_csv(url)

# Display first few rows
print("First 5 rows of the dataset:")
print(titanic.head())

# Basic information
print("\nDataset information:")
print(titanic.info())

# Summary statistics
print("\nSummary statistics:")
print(titanic.describe())

# Check for missing values
print("\nMissing values count:")
print(titanic.isnull().sum())

# Handle missing values (modern pandas approach)
titanic = titanic.assign(Age=titanic['Age'].fillna(titanic['Age'].median()))

# Feature engineering
titanic = titanic.assign(
    FamilySize=titanic['Siblings/Spouses Aboard'] + titanic['Parents/Children Aboard'],
    IsAlone=(titanic['Siblings/Spouses Aboard'] + titanic['Parents/Children Aboard'] == 0).astype(int)
)

# Create age groups
titanic = titanic.assign(
    AgeGroup=pd.cut(titanic['Age'],
                    bins=[0, 12, 18, 30, 50, 100],
                    labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])
)

# Convert categorical variables
label_encoder = LabelEncoder()
titanic = titanic.assign(Sex=label_encoder.fit_transform(titanic['Sex']))

# =============================================
# Gender and Age Analysis Visualizations
# =============================================

plt.figure(figsize=(12, 5))

# Gender survival analysis
plt.subplot(1, 2, 1)
gender_survival = titanic.groupby('Sex', observed=True)['Survived'].mean().reset_index()
gender_survival['Sex'] = ['Male', 'Female']  # Map encoded values to labels
sns.barplot(x='Sex', y='Survived', data=gender_survival)
plt.title('Survival Rate by Gender')
plt.ylabel('Survival Rate')

# Age group survival analysis
plt.subplot(1, 2, 2)
age_survival = titanic.groupby('AgeGroup', observed=True)['Survived'].mean().reset_index()
sns.barplot(x='AgeGroup', y='Survived', data=age_survival,
           order=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])
plt.title('Survival Rate by Age Group')
plt.ylabel('Survival Rate')

plt.tight_layout()
plt.show()

# Pclass and Gender combined analysis
plt.figure(figsize=(10, 6))
sns.barplot(x='Pclass', y='Survived', hue='Sex',
            data=titanic.assign(Sex=titanic['Sex'].replace({0: 'Male', 1: 'Female'})))
plt.title('Survival Rate by Class and Gender')
plt.ylabel('Survival Rate')
plt.show()

# =============================================
# Decision Tree Model
# =============================================

# Select features and target
features = ['Pclass', 'Sex', 'Age', 'Siblings/Spouses Aboard',
            'Parents/Children Aboard', 'Fare', 'FamilySize', 'IsAlone']
X = titanic[features]
y = titanic['Survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train decision tree
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2%}")

# Enhanced confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Death', 'Predicted Survival'],
            yticklabels=['Actual Death', 'Actual Survival'])
plt.title('Confusion Matrix')
plt.show()

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Died', 'Survived']))

# Plot decision tree
plt.figure(figsize=(20, 12))
plot_tree(clf,
         feature_names=features,
         class_names=['Died', 'Survived'],
         filled=True,
         rounded=True,
         proportion=True,
         impurity=False)
plt.title('Decision Tree for Titanic Survival Prediction')
plt.show()

# Feature importance
importances = clf.feature_importances_
feature_importance = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importance['Feature'] = feature_importance['Feature'].replace(
    {'Sex': 'Gender (0=Male,1=Female)'})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance in Survival Prediction')
plt.show()

# =============================================
# Detailed Survival Statistics
# =============================================

print("\nDetailed Survival Statistics:")

# Survival by gender
gender_stats = titanic.groupby('Sex', observed=True)['Survived'].agg(['mean', 'count'])
gender_stats.columns = ['Survival Rate', 'Total Passengers']
gender_stats.index = ['Male', 'Female']
print("\nSurvival by Gender:")
print(gender_stats)

# Survival by age group
age_stats = titanic.groupby('AgeGroup', observed=True)['Survived'].agg(['mean', 'count'])
age_stats.columns = ['Survival Rate', 'Total Passengers']
print("\nSurvival by Age Group:")
print(age_stats)

# Survival by class
class_stats = titanic.groupby('Pclass', observed=True)['Survived'].agg(['mean', 'count'])
class_stats.columns = ['Survival Rate', 'Total Passengers']
print("\nSurvival by Passenger Class:")
print(class_stats)