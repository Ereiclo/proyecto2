import matplotlib.pyplot as plt
import numpy as np

def distance(v1: np.ndarray, v2: np.ndarray, orden: int):
  dim_v1 = len(v1.shape)
  return np.sum(np.abs(v1-v2)**orden,axis=dim_v1-1)**(1/orden)


def generate_matrix_score(X,labels):

    sorted_indexes = labels.argsort()

    a = np.tile(X[sorted_indexes],(len(X),1,1))
    b = X[sorted_indexes].reshape(len(X),1,-1)

    sim_matrix = distance(a,b,2)

    norm_matrix = 1 - (sim_matrix - np.min(sim_matrix))/(np.max(sim_matrix) - np.min(sim_matrix))


    return norm_matrix
