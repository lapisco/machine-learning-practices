from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


iris = load_iris()
X = iris.data
y = iris.target

# Hold-out with balanced division
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25)
# print(sss.get_n_splits(X, y))

for train_index, test_index in sss.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


    indexs, c = np.unique(y_test, return_counts=True)
    print('>> {} {}'.format(c, np.sum(c)))


    # kNN train
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train)

    y_predicted = neigh.predict(X_test)
    acc = np.sum(y_test == y_predicted)/len(y_test)

    print('{}\n{}'.format(y_test, y_predicted))
    print(acc)
