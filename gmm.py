from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.stats import multivariate_normal

def get_new_means(X,probs):
    # producto = np.tensordot(X, probs, axes=(0,0)).T
    # return producto/get_N(probs)

    means = []
    _,k =  probs.shape
    N_i = get_N(probs)

    for i in range(k):
        means.append(np.sum(X*probs[:,i].reshape((-1,1)),axis=0)/N_i[i])
        # print(X*(probs[:,i].reshape(-1,1)))

    return np.array(means)


def get_N(probs):
    return np.sum(probs,axis=0)

def new_cov(X,probs,means):
    nd = []
    n,n_cls = probs.shape 
    N_i = get_N(probs)

    # print(probs.shape)
    

    for i in range(n_cls):

        temp = X - means[i]

        nd.append(((temp.T) @ (temp*probs[:,i].reshape((-1,1))))/N_i[i])

    return nd

def get_pi(probs):
    n,_ = probs.shape

    return get_N(probs)/n

        
def get_probs(X,means,cov,pi):
    c = len(pi) 
    probs = []
    
    for k_i in range(c):
        temp1 = np.array(multivariate_normal.pdf(X, mean=means[k_i], cov=cov[k_i]))
        numerador = pi[k_i]*temp1
        denominador = np.zeros(len(X)) 


        for j in range(c):
            temp = np.array(multivariate_normal.pdf(X, mean=means[j], cov=cov[j]))
            denominador = denominador + pi[j]*temp
        
               
        probs.append(numerador/denominador)
    
    # print(probs[0][0])
    # print(probs[1][0])
    # print(probs[2][0])
    # # print(np.array(probs).reshape((-1,c))[0,:])
    # print(np.array(probs).T[0,:])


    return np.array(probs).T
            


def gmm(X,means,cov,pi,epochs):
    probs = get_probs(X,means,cov,pi)

    # for i in range(10):
        # print(probs[i])

    for i in range(epochs):
        if i % 100 == 0:
            print(f'Estamos en la epoca {i}')

        means = get_new_means(X,probs)
        cov = new_cov(X,probs,means)
        pi = get_pi(probs)
        probs = get_probs(X,means,cov,pi)        

    


    return np.argmax(probs, axis=1)
    

X, y = load_iris(return_X_y=True)
# print(y)
X = X[:,2:]
kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(X)

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

clases = gmm(X,means,cov,pi,1000)

# print(pi)
print(clases)
print(y)
plt.scatter(X.T[0], X.T[1], c=clases)
plt.show()
    






