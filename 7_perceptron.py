from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import numpy as np

def load_iris_3classes():
    data = load_iris()
    return data.data,data.target

def load_iris_binary():
    data = load_iris()
    X = data.data
    y = data.target
    y[y == 2] = 1

    return X,y

def plota_dados(X, weights):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k')

    # x_line = np.linspace(-1,1,15)
    # y_line = weights[0, 1]/x_line

    # for i in range(0, len(weights[0, :])):
    # plt.plot(x_line, y_line)
    #
    plt.show()

if __name__ == '__main__':


    # X, y = load_iris_3classes()
    X, y = load_iris_binary()

    clf = Perceptron(tol=1e-3)
    clf.fit(X, y)

    print(clf.coef_)
    print(clf.intercept_)
    print(clf.predict(X))

    plota_dados(X, clf.coef_)


