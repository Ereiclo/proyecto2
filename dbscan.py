from sklearn.datasets import load_iris
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import numpy as np

def hasLabel(label):
    return label >= 0

def isNoise(label):
    return label == 0



def db_scan(X,r=0.5, min_vecinos = 3):
    tree = KDTree(X)
    eps = .0000000001

    labels = np.full(shape=X.shape[0], fill_value=-1)

    actual_label = 1

    for p_i in range(len(X)):
        if hasLabel(labels[p_i]):
            continue

        [indexes],[_] = tree.query_radius([X[p_i]],r=r + eps,return_distance=True,sort_results=True)

        if len(indexes) < min_vecinos:
            labels[p_i] = 0 
            continue

        labels[p_i] = actual_label

        indexes = indexes[1:]

        i = 0
        while i < len(indexes):
            p_v = indexes[i]

            if isNoise(labels[p_v]):
                labels[p_v] = actual_label 

            if hasLabel(labels[p_v]) :
                # print('pipipi',labels[p_v],p_v)
                i += 1
                continue

            labels[p_v] = actual_label 

            [new_indexes],[_] = tree.query_radius([X[p_v]],r=r+ eps,return_distance=True,sort_results=True)
            # [new_indexes2],[_] = tree.query_radius([X[p_v]],r=r+.0000000001,return_distance=True,sort_results=True)

            # print(X[p_v])
            # print(X[new_indexes])
            # print(new_indexes)
            # print(_)

            if len(new_indexes) < min_vecinos:
                i += 1
                continue


            # print(p_v)
        
            # print(_)

            
            # print(p_v)
            # print(new_indexes)
            # print(_)
            indexes = np.concatenate([indexes, new_indexes[1:]])
            # print(indexes)
            i += 1

        # print({l for l in labels})
        # plt.scatter(X.T[0], X.T[1], c=labels+1)
        # plt.show()

        actual_label += 1
    
    return labels




if __name__ == "__main__":
    X, y = load_iris(return_X_y=True)
    X = np.unique(X.T[2:3+1].T, axis=0)
    labels = db_scan(X)
    # print({l for l in labels})
    plt.scatter(X.T[0], X.T[1], c=labels+1)
    plt.show()






