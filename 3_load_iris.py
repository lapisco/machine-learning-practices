from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


iris = load_iris()
X = iris.data
y = iris.target


plt.figure(1, figsize=(10,5))
plt.clf()


# Plota os dados
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k')
plt.xlabel('Comprimento sepala')
plt.ylabel('Largura sepala')


plt.subplot(1, 2, 2)
plt.scatter(X[:, 2], X[:, 3], c=y, cmap=plt.cm.Set1, edgecolor='k')
plt.xlabel('Comprimento petala')
plt.ylabel('Largura petala')

plt.show()