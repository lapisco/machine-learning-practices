from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
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


if __name__ == '__main__':

    # X, y = load_iris_binary()
    X, y = load_iris_3classes()


    ##### Model pre-loadings:
    mlp_clf = MLPClassifier(solver='adam', learning_rate='adaptive',
                            max_iter=1300, learning_rate_init=5e-04, tol=1e-4)

    svm_rbf_clf = SVC(kernel='rbf')

    # Hyperameter tunning by randomized search:
    # Classifiers definitions:
    classifiers = {'MLP': mlp_clf, 'SVM-RBF': svm_rbf_clf}

    # Define param range for searching:
    param_dist_dict = {'MLP': {"hidden_layer_sizes": list(np.arange(2,500))},
                       'SVM-RBF': {'gamma': [2**i for i in range(-15,3)], 'C': [2**i for i in range(-5,15)]}}


    random_search = dict((k,[]) for k in classifiers.keys())

    for clf in param_dist_dict.keys():
        random_search[clf] = RandomizedSearchCV(classifiers[clf], param_dist_dict[clf], cv=5, n_iter=10, verbose=5, scoring='accuracy')
        random_search[clf].fit(X, y)

        # TODO save the best