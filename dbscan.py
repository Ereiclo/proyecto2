from sklearn.datasets import load_iris
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

X, y = load_iris(return_X_y=True)

X = np.unique(X.T[2:3+1].T, axis=0)
tree = KDTree(X)
min_vecinos = 3 
r = 0.4


labels = np.full(shape=X.shape[0], fill_value=-1)

actual_label = 1

for p_i in range(len(X)):
    if labels[p_i] >= 0:
        continue
    indexes = tree.query_radius([X[p_i]],r=r)[0]

    if len(X[indexes]) < min_vecinos:
        labels[p_i] = 0 
        continue

    labels[p_i] = actual_label


    indexes = indexes[1:]

    i = 0
    while i < len(indexes):
        p_v = indexes[i]

        if labels[p_v] == 0:
            labels[p_v] = actual_label 

        if labels[p_v] > 0:
            i += 1
            continue

        labels[p_v] = actual_label 

        new_indexes = tree.query_radius([X[p_v]],r=r)[0]

        if len(new_indexes) < min_vecinos:
            i += 1
            continue
        
        indexes = np.concatenate([indexes, new_indexes])

        i += 1


    actual_label += 1
    

plt.scatter(X.T[0], X.T[1], c=labels)
plt.show()

