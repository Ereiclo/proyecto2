from sklearn.datasets import load_iris
from kmeans import Init_Centroide,get_cluster,return_new_centroide
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.stats import multivariate_normal


#la idea general en cada funcion es convertir todas las operaciones a un cubo
#y ahi calcular con cada columna de probs

def get_new_means(X,probs):
    

    _,k =  probs.shape
    N_i = get_N(probs)

    X_ = np.tile(X,(k,1,1)) #convertirlo a cubo
    probs_ = probs.T.reshape(k,-1,1) #convertir cada columna a una dimension del cubo

            #multiplicar el cubo por el cubo de columnas
    return np.sum(X_*probs_,axis=1)/N_i.reshape(-1,1)



def get_N(probs):
    return np.sum(probs,axis=0)

#idea parecida que con new means
def new_cov(X,probs,means):
    n,k = probs.shape 
    N_i = get_N(probs)

    X_ = np.tile(X,(k,1,1))
    means_ = means.reshape(k,1,-1)
    temp = X_ - means_
    probs_ = probs.T.reshape(k,-1,1)

    return (np.transpose(temp,axes=(0,2,1)) @ (temp*probs_)) / N_i.reshape(k,-1,1)


def get_pi(probs):
    n,_ = probs.shape

    return get_N(probs)/n

        
def get_probs(X,means,cov,pi):
    c = len(pi) 
    

    probs = [pi[j]*np.array(multivariate_normal.pdf(X,mean=means[j],cov=cov[j],allow_singular=True)) for j in range(c)]
    probs = np.array(probs)


    return (probs / np.sum(probs,axis=0)).T



def get_init_data(X,k=3,option=1, DEBUG = True):

    kmeans,total_labels,means = None,None,None


    if option:
        if DEBUG:
            print('Using kmeans from sklearn')
        kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(X)
        total_labels = kmeans.labels_
        means = kmeans.cluster_centers_
    else:
        if DEBUG:
            print('Using random most distant points')
        centroides = Init_Centroide(X,k)
        total_labels = get_cluster(X,centroides,-1)
        means = return_new_centroide(total_labels,X,k)


    labels = {lb for lb in total_labels}
    cov = []
    pi = []

    for lb in labels:
        indexes = total_labels == lb
        actual_X = X[indexes]

        pi.append(len(actual_X)/len(X))

        cov_ = np.cov(actual_X.T)
        cov.append(cov_)



    return pi,means,cov


def gmm(X,epochs,k=3, DEBUG=True,option=1):


    pi,means,cov = get_init_data(X,k,option, DEBUG) 
    
    probs = get_probs(X,means,cov,pi)

    for i in range(epochs):
        if i % 100 == 0 and DEBUG:
            print(f'Estamos en la epoca {i}')

        means = get_new_means(X,probs)
        cov = new_cov(X,probs,means)
        pi = get_pi(probs)
        probs = get_probs(X,means,cov,pi)        


    return np.argmax(probs, axis=1)
    

if __name__ == '__main__':
    X, y = load_iris(return_X_y=True)
    # print(y)
    X = X[:,2:]
    

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

    import kmeans

    clases = gmm(X,200,k=3,option=1)
    # _,clases = kmeans.kmeans(X,3,0.5,2)

    # print(pi)
    print(clases)
    print(y)
    plt.scatter(X.T[0], X.T[1], c=clases)
    plt.show()
        






