import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def naive_bayes_ex(X, y):
    clf = GaussianNB()
    clf.fit(X, y)

    print('Variancias {}'.format(clf.sigma_))
    print(clf.predict([[-0.8, -1]]))


def qda_ex(X,y):
    clf = QuadraticDiscriminantAnalysis(store_covariance=True)
    clf.fit(X, y)

    print('Matrizes de covariancia {} '.format(clf.covariance_))
    print('Vetores de media {}'.format(clf.means_))

    print(clf.predict([[-0.8, -1]]))

if __name__ == '__main__':


    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    y = np.array([1, 1, 1, 2, 2, 2])


    naive_bayes_ex(X, y)
    qda_ex(X,y)

