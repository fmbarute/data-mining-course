from sklearn.neighbors import KNeighborsClassifier


def knn_prediction():
    X = [[0, 3, 0], [2, 0, 0], [0, 1, 3], [0, 1, 2], [-1, 0, 1], [1, 1, 1]]
    y = ['Red'] * 3 + ['Green'] * 2 + ['Red']

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)

    print(f"K=3 prediction: {knn.predict([[0, 0, 0]])[0]}")