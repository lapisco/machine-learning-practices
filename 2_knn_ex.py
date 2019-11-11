from sklearn.neighbors import KNeighborsClassifier

X = [[0], [1], [2], [3]] # 1D (4 amostras)
y = [0, 0, 1, 1]

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)

print(neigh.predict([[0.3]]))
print(neigh.predict([[2.1]]))