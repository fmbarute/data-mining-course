import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# Load data with correct path
try:
    data_path = os.path.join(os.path.dirname(__file__), '../Data/Default.csv')
    default = pd.read_csv(data_path)

    # Data preprocessing
    default['default'] = (default['default'] == 'Yes').astype(int)
    default['student'] = (default['student'] == 'Yes').astype(int)

    # Prepare data
    X = default[['balance', 'income', 'student']]
    y = default['default']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Logistic Regression
    logreg = LogisticRegression()
    logreg.fit(X_train_scaled, y_train)
    print("Logistic Regression Performance:")
    print(classification_report(y_test, logreg.predict(X_test_scaled)))

    # Neural Network
    mlp = MLPClassifier(hidden_layer_sizes=(10,), activation='relu',
                        alpha=0.1, dropout=0.2, random_state=42)
    mlp.fit(X_train_scaled, y_train)
    print("\nNeural Network Performance:")
    print(classification_report(y_test, mlp.predict(X_test_scaled)))

except FileNotFoundError:
    print("Error: Could not find Default.csv file.")
    print("Please ensure the file exists in the ../Data/ directory relative to this script.")
except Exception as e:
    print(f"An error occurred: {str(e)}")