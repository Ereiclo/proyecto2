from sklearn.datasets import load_iris
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def hasLabel(label):
    return label >= 0

X, y = load_iris(return_X_y=True)

# X = np.unique(X, axis=0)
X = np.unique(X.T[2:3+1].T, axis=0)
tree = KDTree(X)
min_vecinos = 3 
r = 0.6


labels = np.full(shape=X.shape[0], fill_value=-1)

actual_label = 1

for p_i in range(len(X)):
    if hasLabel(labels[p_i]):
        continue

    # indexes = tree.query_radius([X[p_i]],r=r)[0]
    # print(X[p_i].reshape(1,-1).shape)
    [indexes],[_] = tree.query_radius([X[p_i]],r=r,return_distance=True,sort_results=True)
    # print(p_i)
    # print(indexes)
    # print(_)

    # print(p_i)
    # print(tree.query_radius([X[p_i]],r=r,return_distance=True,sort_results=True))
    # break
    # if p_i < 10:
    #     continue
    # else:
    #     break


    if len(indexes) < min_vecinos:
        labels[p_i] = 0 
        continue

    labels[p_i] = actual_label

    indexes = indexes[1:]

    i = 0
    while i < len(indexes):
        p_v = indexes[i]
        print(indexes,p_v)

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
    
    break


    actual_label += 1
    

# plt.scatter(X.T[0], X.T[1], c=labels)
# plt.show()

