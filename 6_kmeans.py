from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


def plot_dados(X, kmeans):

    colors = [[0, .9, 0], [0, 0, .9]]
    plt.figure(1, figsize=(10,5))
    plt.clf()


    # kmeans.labels_
    # kmeans.predict([[0, 0], [12, 3]])
    clusters = kmeans.cluster_centers_

    # Plota os dados
    plt.subplot(1, 2, 1)
    plt.plot(X[:, 0], X[:, 1], 'o')

    plt.subplot(1, 2, 2)
    for k in range(0, 2):
        plt.plot(X[kmeans.labels_ == k, 0], X[kmeans.labels_ == k, 1], 'o', color=colors[k])
        plt.plot(clusters[k, 0], clusters[k, 1], 'rX', markersize=10)


    plt.show()




if __name__ == '__main__':

    # load dataset
    iris = load_iris()
    X = iris.data

    kmeans = KMeans(n_clusters=2).fit(X)


    print('Predict {} {}'.format(kmeans.predict([X[0,:]]), kmeans.predict([X[140,:]])))
    print('Labes {}'.format(kmeans.labels_))

    plot_dados(X, kmeans)