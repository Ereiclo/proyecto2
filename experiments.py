from  skimage.io import imread, imshow
import pywt
import pywt.data
import matplotlib.pyplot as plt
import numpy as np
import glob
import gmm
from tqdm import tqdm
import json
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


ruta_drive = './matrices_imagenes/'
rutas_pca = glob.glob(ruta_drive + "*.npy")
diccionario_pca = {}
for ruta in rutas_pca:
  if len(ruta.split('_')) > 1:
    diccionario_pca[os.path.splitext(os.path.basename(ruta))[0]] = np.load(ruta)

# print(diccionario_pca.keys())


X = diccionario_pca['caracteristicos2_pca0.99']

kmeans = KMeans(n_clusters=8, random_state=0, n_init="auto").fit(X)

total_labels = kmeans.labels_

labels = {lb for lb in kmeans.labels_}
# print(kmeans.cluster_centers_)
# print(labels)


means = kmeans.cluster_centers_
# cov = [np.eye(len(X),len(X)) for _ in range(len(labels))]
cov = []
pi = []

for lb in labels:
    indexes = total_labels == lb
    actual_X = X[indexes]

    pi.append(len(actual_X)/len(X))

    cov_ = np.cov(actual_X.T)
    cov.append(cov_)

    # for i in range(len(cov)):
        # cov_[i][i] = 1


    # print(cov_.shape)


# cov = np.concatenate(cov)
# pi = np.array(pi)
# print(X.shape)

# probs = get_probs(X,means,cov,pi)


# get_new_means(X,probs)
# print(get_probs(X,means,cov,pi).shape)
# for gamma_i in get_probs(X,means,cov,pi):
    # print(gamma_i.shape)
# print(cov)

# print(kmeans.cluster_centers_)

print(X.shape)
print(np.array(means).shape)
print(np.array(cov).shape)
print(np.array(pi).shape)
clases = gmm.gmm(X,means,cov,pi,1000)