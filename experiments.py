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
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


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

class Experimentos():
    def K_Medias():
        k = range(2, 10 + 1)
        umbral = [0.5, 0.1, 0.05, 0.01]
        orden = [1, 2, -1]
        for archivo in dicc_pca:
            resultados = {
                "k": [],
                'umbral': [],
                'orden': [],
                'silhouette_score': [],
                'davies_bouldin_score': [],
                'calinski_harabasz_score': []
            }
            X = dicc_pca[archivo]
            for k_i in tqdm(k):
                for umbral_i in umbral:
                    for orden_i in orden:
                        _, clases = kmeans.kmeans(data=X, k=k_i, umbral=umbral_i, orden=orden_i, DEBUG=False)
                        resultados['k'].append(k_i)
                        resultados['umbral'].append(umbral_i)
                        resultados['orden'].append(orden_i)
                        resultados['silhouette_score'].append(silhouette_score(X,clases))
                        resultados['davies_bouldin_score'].append(davies_bouldin_score(X,clases))
                        resultados['calinski_harabasz_score'].append(calinski_harabasz_score(X,clases))

            pd.DataFrame(resultados).to_csv(f'resultados_experimentos/k_medias/{archivo}.csv',index=False)

    def GMM():
        k = range(2, 10+1)
        epoca = [100, 500, 1000]
        contador_por_error = 0
        for archivo in dicc_pca:
            if contador_por_error < 6:
                contador_por_error+=1
                continue
            print(archivo)
            resultados = {
                "k": [],
                'epoca': [],
                'silhouette_score': [],
                'davies_bouldin_score': [],
                'calinski_harabasz_score': []
            }
            X = dicc_pca[archivo]
            for k_i in tqdm(k):
                for epoca_i in epoca:
                    clases = gmm.gmm(X=X, epochs=epoca_i, k=k_i, DEBUG=False)
                    resultados['k'].append(k_i)
                    resultados['epoca'].append(epoca_i)
                    resultados['silhouette_score'].append(silhouette_score(X,clases))
                    resultados['davies_bouldin_score'].append(davies_bouldin_score(X,clases))
                    resultados['calinski_harabasz_score'].append(calinski_harabasz_score(X,clases))

            pd.DataFrame(resultados).to_csv(f'resultados_experimentos/gmm/{archivo}.csv',index=False)
            



    def DBSCAN():
        umbral = [0.5, 0.1, 0.05, 0.01]
        resultados = {
            'archivo': [],
            'n_clases': [],
            'umbral': [],
            'silhouette_score': [],
            'davies_bouldin_score': [],
            'calinski_harabasz_score': []
        }
        for archivo in dicc_pca:
            X = dicc_pca[archivo]
            print(X.shape)
            for umbral_i in umbral:
                clases = dbscan.db_scan(X, umbral_i)
                print(np.unique(clases).size)
                resultados['archivo'].append(archivo)
                resultados['n_clases'].append(np.unique(clases).size)
                resultados['umbral'].append(umbral_i)
                resultados['silhouette_score'].append(silhouette_score(X,clases))
                resultados['davies_bouldin_score'].append(davies_bouldin_score(X,clases))
                resultados['calinski_harabasz_score'].append(calinski_harabasz_score(X,clases))

        pd.DataFrame(resultados).to_csv(f'resultados_experimentos/dbscan/resultados.csv',index=False)



Experimentos.GMM()

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

