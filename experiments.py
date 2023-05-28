from  skimage.io import imread, imshow
import kmeans
from sklearn.metrics import davies_bouldin_score,silhouette_score,calinski_harabasz_score
from sim_matrix import generate_matrix_score
import dbscan
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



def load_data():
    ruta_drive = './matrices_imagenes/'
    rutas_pca = glob.glob(ruta_drive + "*.npy")
    diccionario_pca = {}
    for ruta in rutas_pca:
        nombre_archivo = os.path.basename(ruta)
        if len(nombre_archivo.split('_')) != 1:
            diccionario_pca[os.path.splitext(nombre_archivo)[0]] = np.load(ruta)
    
    return diccionario_pca


dicc_pca = load_data()


""" 
X = dicc_pca['caracteristicos2_pca0.99']

# clases = dbscan.db_scan(X,500)
_,clases = kmeans.kmeans(X,8,0.5,2)
# clases = gmm.gmm(X,100,k=8)
matrix_score = generate_matrix_score(X,clases)


print(silhouette_score(X,clases))
print(davies_bouldin_score(X,clases))
print(calinski_harabasz_score(X,clases))
# plt.imshow(matrix_score,cmap='seismic')
# plt.show() """

