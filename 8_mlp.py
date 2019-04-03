from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np

def decision_surface_mlp(n_classes):
    plot_colors = "ryb"
    plot_step = 0.02

    # Load data
    iris = load_iris()

    target_names = iris.target_names
    if n_classes == 2:
        target_names = ['setosa', 'others']
    for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3],
                                    [1, 2], [1, 3], [2, 3]]):
        # We only take the two corresponding features
        X = iris.data[:, pair]
        y = iris.target
        if n_classes == 2:
            y[y == 2] = 1

        # Train
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(80,), random_state=1)
        clf.fit(X, y)

        # Plot the decision boundary
        plt.subplot(2, 3, pairidx + 1)

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                             np.arange(y_min, y_max, plot_step))
        plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

        plt.xlabel(iris.feature_names[pair[0]])
        plt.ylabel(iris.feature_names[pair[1]])

        # Plot the training points
        for i, color in zip(range(n_classes), plot_colors):
            idx = np.where(y == i)
            plt.scatter(X[idx, 0], X[idx, 1], c=color, label=target_names[i],
                        cmap=plt.cm.RdYlBu, edgecolor='black', s=25)

    plt.suptitle("Decision surface of a MLP using paired features")
    plt.legend(loc='lower right', borderpad=0, handletextpad=0)
    plt.axis("tight")
    plt.show()


def decision_surface_mlp_xor():
    plot_step = 0.02

    # Load data
    X = [[0., 0.], [1., 1.], [0., 1.], [1., 0.]]
    y = [0, 0, 1, 1]


    # Train
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10,), random_state=1)
    clf.fit(X, y)

    # Plot the decision boundary

    x_min, x_max = -0.1, 1.1
    y_min, y_max = -0.1, 1.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)


    plt.plot(0, 0, 'ko', markersize=20)
    plt.plot(1, 1, 'ko', markersize=20)
    plt.plot(1, 0, 'wx', markersize=20)
    plt.plot(0, 1, 'wx', markersize=20)

    plt.suptitle("Decision surface of a MLP for xor problem")
    plt.legend(loc='lower right', borderpad=0, handletextpad=0)
    plt.axis("tight")
    plt.show()

if __name__ == '__main__':

    X = [[0., 0.], [1., 1.], [0., 1.], [1., 0.]]
    y = [0, 0, 1, 1]

    # max_iter=1300, learning_rate_init=5e-04, tol=1e-4
    clf = MLPClassifier(hidden_layer_sizes=(10,),
                        learning_rate='adaptive', max_iter=5000, learning_rate_init=5e-04, tol=1e-4)
    # clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (15,), random_state = 1)
    clf.fit(X, y)
    print(clf.predict(X))



    decision_surface_mlp_xor()


    decision_surface_mlp(3)

