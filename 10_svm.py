from sklearn.svm import SVC
import numpy as np

X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])


# SVM
clf = SVC(kernel='rbf', gamma=2, C=4)
clf.fit(X, y)


print(clf.predict([[-0.8, -1]]))

# TODO Carregar a base de dados Iris
# TODO Plotar as superficies de decisao
# TODO Calcular as acurácias
