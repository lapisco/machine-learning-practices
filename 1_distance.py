from scipy.spatial import distance

# X1. esférico.
# X2. sim.
# X3. alaranjado.
# X4. rugosa.
# X5. sim.

laranja     = [0, 1, 2, 1, 0]
maca        = [0, 0, 1, 0, 0]

tangerina   = [0, 1, 2, 1, 1]


print(distance.cityblock(laranja, tangerina))
# print(distance.euclidean(laranja, tangerina))

# print(distance.euclidean(maca, tangerina))
print(distance.cityblock(maca, tangerina))
