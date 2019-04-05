from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def load_iris_3classes():
    data = load_iris()
    return data.data,data.target

def load_iris_binary():
    data = load_iris()
    X = data.data
    y = data.target
    y[y == 2] = 1

    return X,y


def naive_bayes_ex(X, y):
    clf = GaussianNB()
    clf.fit(X, y)

    print('\n---------Naive Bayes----------')
    for i, sigma in enumerate(clf.sigma_):
        print('Variancia da classe {}\n {}'.format(i, sigma))
    print(clf.predict([[4.9, 3.1, 1.5, 0.1]]))


def qda_ex(X,y):
    clf = QuadraticDiscriminantAnalysis(store_covariance=True)
    clf.fit(X, y)

    print('\n-------------QDA-------------')

    for i,cov in enumerate(clf.covariance_):
        print('Matriz de covariancia da classe {} \n {}'.format(i, cov))

    for i, mean in enumerate(clf.means_):
        print('Vetor de media da classe {} \n {}'.format(i, mean))

    print(clf.predict([[5.1, 3.5, 1.4, 0.2]]))

def lda_ex(X,y):
    clf = LinearDiscriminantAnalysis(store_covariance=True)
    clf.fit(X, y)

    print('\n-------------LDA-------------')
    print('Matriz de covariancia das classes \n{}'.format(clf.covariance_))

    for i, mean in enumerate(clf.means_):
        print('Vetor de media da classe {} \n {}'.format(i, mean))

    print(clf.predict([[5.1, 3.5, 1.4, 0.2]]))

if __name__ == '__main__':
    X, y = load_iris_binary()
    # X, y = load_iris_3classes()

    qda_ex(X,y)
    naive_bayes_ex(X, y)
    lda_ex(X, y)

    # TODO Plotar as superficies de decisao
    # TODO Calcular as acurácias
