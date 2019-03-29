from scipy.spatial import distance

setosa      = [5.1, 3.5, 1.4, 0.2]
versicolor  = [6.5, 2.8, 4.6, 1.5]
virginica   = [7.1, 3.0, 5.9, 2.1]

print(distance.cityblock(setosa, versicolor))
print(distance.cityblock(setosa, virginica))

print(distance.euclidean(setosa, versicolor))
print(distance.euclidean(setosa, virginica))

