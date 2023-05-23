import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.pyplot import imread
from matplotlib.pyplot import imshow
import numpy as np


# Devuelve la distancia entre 2 vectores.
# Pruebe con varias funciones de distancia.
def distance(v1: np.ndarray, v2: np.ndarray, orden: int):
  # dif = np.abs(v1-v2)**orden
  # return np.sum(dif)**(1/orden)
  dim_v1 = len(v1.shape)
  return np.sum(np.abs(v1-v2)**orden,axis=dim_v1-1)**(1/orden)

# Inicialice los k centroides tomando aleatoriamente k elementos de los datos para que cada
# centroide tenga una ubicación inicial en el espacio de características.


def Init_Centroide(data: np.ndarray, k: int):
  indices_aleatorios = np.random.choice(len(data), k, replace=False)

  return data[indices_aleatorios]

# Dado que los grupos se han formado previamente, se pueden obtener nuevos
# centroides calculando el vector promedio de cada grupo.
def return_new_centroide(grupos: np.ndarray, data: np.ndarray, k: int):
  
  # nuevos_centroides = []
  # for k_i in range(k):
  #   indices_clase_k_i = grupos == k_i
  #   cluster = data[indices_clase_k_i]
  #   nuevos_centroides.append(np.mean(cluster,axis=0))

    
  # return np.array(nuevos_centroides)
  data_ = np.tile(data,(k,1,1))
  labels = [i for i in range(k)]
  grupos_matrix = np.tile(grupos.reshape(-1,1),(1,k)) == labels
  grupos_ = grupos_matrix.T.reshape(k,len(data),1)
  temp = data_*grupos_
  
  return np.sum(temp,axis=1)/np.sum(grupos_matrix,axis=0).reshape(-1,1)


  # return np.sum(temp,axis=1)/(np.sum(temp > 0,axis=(1,2))/data.shape[1]).reshape(-1,1)

  


# La función devuelve un vector de números entre 0 y k-1, donde cada valor indica la clase
# a la que pertenece cada elemento del dataset.


def get_cluster(data: np.ndarray, centroides: np.ndarray, orden: int):
  # distancia_centroides = np.array([distance(data, centroide, orden).reshape(-1,1)
  #                         for centroide in centroides])

  # print(centroides)
  k = len(centroides)
  data_ = np.tile(data,(k,1,1))
  centroides_ = centroides.reshape(k,1,-1)
  distancia_por_centroide = distance(data_,centroides_,orden)

  return np.argmin(distancia_por_centroide,axis=0)


# Halla la distancia promedio entre los antiguos centroides y los nuevos centroides para evaluar
# la convergencia del algoritmo K-Means y determinar si se debe continuar con la iteración.


def distancia_promedio_centroides(old_centroide: np.ndarray, new_centroide: np.ndarray, orden: int):
  # distancias = [distance(antiguo, nuevo, orden)
                # for antiguo, nuevo in list(zip(old_centroide, new_centroide))]
  distancias = distance(old_centroide,new_centroide,orden)
  return np.mean(distancias)

# Este es el algoritmo K-Means. Debe retornar los centroides y los clusters
# generados para poder utilizarlos en análisis posteriores.


def kmeans(data: np.ndarray, k: int, umbral, orden):
  centroides = Init_Centroide(data, k)
  # print(centroides)
  print("CENTROIDE LISTO")
  clusters = get_cluster(data, centroides, orden)

  print(Counter(clusters))
  print(Counter(clusters).total())
  print(len(data))
  print("CLUSTERS LISTOS")
  new_centroides = return_new_centroide(clusters, data, k)
  print("NUEVOS CENTROIDES")
  _iter = 0
  # print(new_centroides)
  while (distancia_promedio_centroides(centroides, new_centroides, orden) > umbral):

     _iter += 1
     centroides = new_centroides
     clusters = get_cluster(data, centroides, orden)
     new_centroides = return_new_centroide(clusters, data, k)

     print(_iter, distancia_promedio_centroides(centroides, new_centroides, orden))

  return new_centroides, clusters

# A cada imagen le corresponde un único valor de cluster, y a cada cluster se le asigna un color. Luego, grafique la imagen
# recoloreando los píxeles según el cluster al que pertenecen.


def Show_Imagen():

  np.random.seed(34)

  img = imread("800wm.jpeg")
  img_shape = img.shape
  img = img.reshape(-1, img.shape[-1])  # [[] []  [] [} ]]
  k = 3 
  umbral = 0.5 
#   print(img.shape)
#   img_sample =  np.random.choice(img.shape[0],1000, replace=False)
#   print(img_sample.shape)
  # print(img.shape)
  # print(np.unique(img,axis=0).shape)
  new_img = np.zeros(img.shape, dtype=img.dtype)
  centroides, clases = kmeans(data=np.unique(img,axis=0), k=k, umbral=umbral, orden=2)


  complete_cluster = get_cluster(img,centroides,2)

  # _, clases = kmeans(img, k, 2,2)

  colors = np.random.randint(0, 256, size=(k, 3))

  for k_i in range(k):
    # indices_clase_k_i = np.where(clases == k_i)
    # pixels_from_cluster = img.take(indices_clase_k_i)
    new_img[complete_cluster == k_i] = colors[k_i]

  new_img = new_img.reshape(img_shape)
  # [] [] []
  # [] [] []
  # [] [] []
  imshow(new_img)
  plt.show()


Show_Imagen()
