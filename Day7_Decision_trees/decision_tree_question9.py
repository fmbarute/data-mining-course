import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from ISLP import load_data

# (a) Load and split data
OJ = load_data('OJ')
X = OJ.drop(['Purchase'], axis=1)
y = OJ['Purchase']

# Convert categorical variables to dummy variables
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=800, random_state=42)

# (b) Fit initial tree
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train, y_train)
train_err = 1 - tree.score(X_train, y_train)
print(f"(b) Training Error Rate: {train_err:.3f}")

# (c) Plot tree
plt.figure(figsize=(20,10))
plot_tree(tree, filled=True, feature_names=X.columns, class_names=['CH', 'MM'], rounded=True)
plt.show()
n_leaves = tree.get_n_leaves()
print(f"(c) Number of Terminal Nodes: {n_leaves}")

# (d) Text summary of tree
tree_text = export_text(tree, feature_names=list(X.columns))
print("(d) Tree Summary (excerpt):")
print(tree_text.split('\n')[0])  # Print first split only

# (e) Test predictions
y_pred = tree.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
test_err = 1 - accuracy_score(y_test, y_pred)
print("\n(e) Confusion Matrix:")
print(cm)
print(f"Test Error Rate: {test_err:.3f}")

# (f)-(h) Cross-validation for optimal tree size
max_depths = range(1, 21)
cv_scores = []

for depth in max_depths:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    scores = cross_val_score(tree, X_train, y_train, cv=5)
    cv_scores.append(1 - np.mean(scores))

optimal_depth = max_depths[np.argmin(cv_scores)]
print(f"\n(h) Optimal Tree Depth: {optimal_depth}")

# (g) Plot CV error vs tree size
plt.figure(figsize=(10,6))
plt.plot(max_depths, cv_scores, 'bo-')
plt.xlabel('Tree Depth')
plt.ylabel('Cross-Validated Error Rate')
plt.title('Tree Complexity vs Error Rate')
plt.axvline(x=optimal_depth, color='red', linestyle='--')
plt.show()

# (i) Pruned tree
pruned_tree = DecisionTreeClassifier(max_depth=optimal_depth, random_state=42)
pruned_tree.fit(X_train, y_train)

# If no pruning from CV (unlikely), force 5 terminal nodes
if optimal_depth == max(max_depths):
    pruned_tree = DecisionTreeClassifier(max_leaf_nodes=5, random_state=42)
    pruned_tree.fit(X_train, y_train)

# (j) Compare training errors
pruned_train_err = 1 - pruned_tree.score(X_train, y_train)
print(f"\n(j) Training Error Rates:")
print(f"Unpruned: {train_err:.3f}, Pruned: {pruned_train_err:.3f}")

# (k) Compare test errors
pruned_pred = pruned_tree.predict(X_test)
pruned_test_err = 1 - accuracy_score(y_test, pruned_pred)
print(f"\n(k) Test Error Rates:")
print(f"Unpruned: {test_err:.3f}, Pruned: {pruned_test_err:.3f}")