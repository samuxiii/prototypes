import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

X = np.array([[1, 2], [4, 4], [2, 1], [1, 3], [3, 2], [1, 4], [2, 3]])

n_neighbors = 3
nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(X)

distances, indices = nbrs.kneighbors(X)

plt.figure()
plt.scatter(X[:, 0],X[:, 1])
plt.show()

#new point to calculate neighbors
coorx = input("Define coordinate x: ")
coory = input("Define coordinate y: ")

new = np.array([coorx, coory]).reshape(1, -1)
newDist, newInd = nbrs.kneighbors(new)
print "Neighbors indices: "
print newInd

plt.figure()
plt.scatter(X[:, 0],X[:, 1])
plt.plot(X[newInd, 0], X[newInd, 1], 'yo')
print new
plt.plot(new[:, 0], new[:, 1], 'ro')
plt.show()
