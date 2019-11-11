from scipy.spatial import distance

# X1. Formato (esférico/oval)?
# X2. Fruta cítrica?
# X3. Cor?
# X4. Casca lisa ou rugosa?
# X5. Cheiro ativo?


laranja     = [0, 1, 2, 1, 0]
maca        = [0, 0, 1, 0, 0]

tangerina   = [0, 1, 2, 1, 1]


print("Cityblock (laranja, tangerina): {}".format(distance.cityblock(laranja, tangerina)))
print("Euclidean (laranja, tangerina): {}".format(distance.euclidean(laranja, tangerina)))

print("Cityblock (maca, tangerina): {}".format(distance.cityblock(maca, tangerina)))
print("Euclidean (maca, tangerina): {}".format(distance.euclidean(maca, tangerina)))
