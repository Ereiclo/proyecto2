from matplotlib.pyplot import imread
from matplotlib.pyplot import imshow
import numpy as np


# Devuelve la distancia entre 2 vectores.
# Pruebe con varias funciones de distancia.
def distance(v1: np.ndarray, v2: np.ndarray, orden: int):
  dif = np.abs(v1-v2)**orden
  return np.sum(dif)**(1/orden)

# Inicialice los k centroides tomando aleatoriamente k elementos de los datos para que cada
# centroide tenga una ubicación inicial en el espacio de características.


def Init_Centroide(data: np.ndarray, k: int):
  indices_aleatorios = np.random.choice(data.size, k, replace=False)
  return data.take(indices_aleatorios)

# Dado que los grupos se han formado previamente, se pueden obtener nuevos
# centroides calculando el vector promedio de cada grupo.


def return_new_centroide(grupos: np.ndarray, data: np.ndarray, k: int):
  nuevos_centroides = []
  for k_i in range(k):
    indices_clase_k_i = np.where(grupos == k_i)[0]
    nuevos_centroides.append(np.mean(data.take(indices_clase_k_i), axis=0))
  return np.array(nuevos_centroides)

# La función devuelve un vector de números entre 0 y k-1, donde cada valor indica la clase
# a la que pertenece cada elemento del dataset.


def get_cluster(data: np.ndarray, centroides: np.ndarray, orden: int):
  distancia_centroides = np.array([distance(data, centroide, orden).reshape(-1,1)
                          for centroide in centroides])
  return np.argmin(distancia_centroides,axis=1)


# Halla la distancia promedio entre los antiguos centroides y los nuevos centroides para evaluar
# la convergencia del algoritmo K-Means y determinar si se debe continuar con la iteración.


def distancia_promedio_centroides(old_centroide: np.ndarray, new_centroide: np.ndarray, orden: int):
  distancias = [distance(antiguo, nuevo, orden)
                for antiguo, nuevo in list(zip(old_centroide, new_centroide))]
  return np.mean(distancias)

# Este es el algoritmo K-Means. Debe retornar los centroides y los clusters
# generados para poder utilizarlos en análisis posteriores.


def kmeans(data: np.ndarray, k: int, umbral, orden):
  centroides = Init_Centroide(data, k)
  print("CENTROIDE LISTO")
  clusters = get_cluster(data, centroides, orden)
  print("CLUSTERS LISTOS")
  new_centroides = return_new_centroide(clusters, data, k)
  print("NUEVOS CENTROIDES")
  _iter = 0
  while (distancia_promedio_centroides(centroides, new_centroides, orden) > umbral):
     print(_iter, distancia_promedio_centroides(
         centroides, new_centroides, orden))
     _iter += 1
     centroides = new_centroides
     clusters = get_cluster(data, centroides, orden)
     new_centroides = return_new_centroide(clusters, data, k)

  return new_centroides, clusters

# A cada imagen le corresponde un único valor de cluster, y a cada cluster se le asigna un color. Luego, grafique la imagen
# recoloreando los píxeles según el cluster al que pertenecen.


def Show_Imagen():

  img = imread("800wm.jpg")
  img_shape = img.shape
  img = img.reshape(-1, img.shape[-1])  # [[] []  [] [} ]]
  k = 3
  umbral = 4.5
#   print(img.shape)
#   img_sample =  np.random.choice(img.shape[0],1000, replace=False)
#   print(img_sample.shape)
  new_img = np.zeros(img.shape, dtype=img.dtype)
  _, clases = kmeans(data=img, k=k, umbral=umbral, orden=2)
#   _, clases = kmeans(img, k, 2,2)

  colors = np.random.randint(0, 256, size=(k, 3))

  for k_i in range(k):
    indices_clase_k_i = np.where(clases == k_i)
    # pixels_from_cluster = img.take(indices_clase_k_i)
    new_img[indices_clase_k_i] = colors[k_i]

  new_img = new_img.reshape(img_shape)
  # [] [] []
  # [] [] []
  # [] [] []
  imshow(new_img)


Show_Imagen()
