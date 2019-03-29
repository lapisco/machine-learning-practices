from scipy.spatial import distance

# X1. Formato (esférico/oval)?
# X2. Fruta cítrica?
# X3. Cor?
# X4. Casca lisa ou rugosa?
# X5. Cheiro ativo?


laranja     = [0, 1, 2, 1, 0]
maca        = [0, 0, 1, 0, 0]

tangerina   = [0, 1, 2, 1, 1]


print(distance.cityblock(laranja, tangerina))
print(distance.euclidean(laranja, tangerina))

print(distance.euclidean(maca, tangerina))
print(distance.cityblock(maca, tangerina))
