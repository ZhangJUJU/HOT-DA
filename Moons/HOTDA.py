import numpy as np
from scipy.spatial import distance
import ot
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn import datasets
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.gaussian_process.kernels import RBF
from sklearn.cluster import SpectralClustering
import numpy as np
import cv2,os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import glob
from matplotlib import pyplot
import seaborn as sns
import warnings
from sklearn.manifold import TSNE

"""
Discover target measures
"""

DBL_MAX = np.finfo('float').max
DBL_MIN = np.finfo('float').min

def algo0(a, b, M, epsilon = 0.1, param='primal', max_iter=50):
    lamda = 1/epsilon
    n = M.shape[0]
    l_b = M.shape[1]
    K = np.zeros((n, l_b))
    K_til = np.zeros((n, l_b))
    
    for i in range(l_b):
        for j in range(n):
            tmp = np.exp(-lamda*M[j,i])
            K[j, i] = tmp
            tmp = tmp/a[j]
            if np.isinf(tmp) or np.isnan(tmp):
                K_til[j, i] = DBL_MAX
            else:
                K_til[j, i] = tmp
    
    it = 0
    u = np.ones(n)/n
    temp_v = np.zeros(l_b)
    
    while it < max_iter:
        
        for i in range(l_b):
            tmp = 0
            for j in range(n):
                tmp = tmp + K[j,i]*u[j]
            tmp = b[i]/tmp
            if np.isinf(tmp):
                temp_v[i] = DBL_MAX
            elif np.isnan(tmp):
                temp_v[i] = 0
            else:    
                temp_v[i] = tmp # check for zero
        
        for j in range(n):
            tmp = 0
            for i in range(l_b):
                tmp = tmp + K_til[j,i] * temp_v[i]
            if tmp < DBL_MIN:
                u[j] = DBL_MAX
            else:
                u[j] = 1/tmp # check for zero

        it = it + 1
    
    t = np.zeros((n, l_b))
    
    if param=='primal':
        for i in range(l_b):
            for j in range(n):
                t[j,i] = K[j,i] * u[j] * temp_v[i]
        return t
    
    else: # param=='dual'
        alpha = np.zeros(n)
        tmp = 0
        for j in range(n):
            if u[j]!=0:
                u[j] = np.log(u[j])
                tmp = tmp + u[j]
        tmp = tmp/(lamda*n)
        for j in range(n):
            alpha[j] = +(tmp - u[j]/lamda)
        return alpha
def algo1(X, Y, b, M, weight=None, max_iter=[10,50]):
    d,n = X.shape
    N = len(Y)
    
    # Initializing importance weights and weights of barycenter unless provided
    if weight is None:
        weight = np.repeat(1./N, N)
        
    a_hat = a_til = np.ones(n)/n
    t = t_0 = 1
    
    while t< max_iter[0]:
        beta = (t+1)/2
        a = (1-(1/beta))*a_hat+(1/beta)*a_til
        alpha_list = [algo0(a, b[i], M[i], param='dual', 
                            max_iter=max_iter[1]) for i in range(N)]
        alpha = [weight[i]*alpha_list[i] for i in range(N)]
        alpha = np.sum(alpha, axis=0)
        
        a_til_n = a_til * np.exp(-t_0*beta*alpha)
        
        # Solving potential numeric issues
        if np.sum(np.isinf(a_til_n)) == 1:
            a_til = np.zeros((n,))
            a_til[np.isinf(a_til_n)] = 1.
        elif np.all(a_til_n==0):
            a_til = np.ones((n,))/n
        else:
            a_til = a_til_n/a_til_n.sum()
        
        a_hat = (1-1/beta)*a_hat + a_til/beta
        if np.any(np.isnan(a_hat)):
            print('Something is wrong in Algo1 Cuturi')
        t = t+1
    return a_hat
def algo2(Y, b, n, weight=None, max_iter=[5, 10, 50]):
    N = len(Y)
    d = Y[0].shape[0]    
    # Initializing importance weights, atoms of barycenter and 
    # weights of barycenter unless provided
    if weight is None:
        weight = np.repeat(1./N, N)
    #X = np.random.normal(3, 5, (d,n))
    #X=Z.T
    tmp_Y0=Y[0].T.copy()
    np.random.shuffle(tmp_Y0)
    X=tmp_Y0.T[:,:n]
    
    
    a = np.ones(n)/n
    t = 1
    
    while t < max_iter[0]:
        print("\n[iter] :",t)
        teta = 3/4
        M = [cdist(X.T,Y[i].T, metric='sqeuclidean') for i in range(N)]
        a = algo1(X, Y, b, M, weight=weight, max_iter=max_iter[1:])
        print("[a] -------------\n",a)
        T_list = [algo0(a, b[i], M[i], max_iter=max_iter[2]) for i in range(N)]
        #print("T -------------\n",T_list[0])
        g = [weight[i]*np.dot(Y[i],T_list[i].T) for i in range(N)]
        g = np.sum(g, axis=0)/a[None,:]
        X = (1-teta)*X + teta*g
        #np.add(X,0.1*np.reshape(np.random.randn(X.size),X.shape))
        print("[X] -------------\n",X.T)
        #plot(X,Y)
        t = t+1
        if np.any(np.isnan(X)):
            print('Something is wrong in Algo2 Cuturi')
    
    return X, a


def Source_target_processing(X,y):  # y must be an np.array and not a list.
    S=[]
    a=[]
    mu=[]
    yc_source=[]
    classes=np.unique(y)
    k=len(classes)
    for i in range(k):#parcourir les classes
        C=X[y==i]
        yc_source=yc_source+list(y[y==i])
        w=np.ones(C.shape[0])/C.shape[0]#1/n
        S.append(C)#stocker les classes
        a.append(w)
        mu.append(C.shape[0]/X.shape[0])#nb_donner_class/nb_total
    mu=np.array(mu)
    return S,a,mu,yc_source

def Hot(S,a,mu,T,b,nu,reg1,reg2):  
    W=np.zeros((len(S),len(T)))
    for i in range(len(S)):
        for j in range(len(T)):
            M=distance.cdist(S[i],T[j], metric='sqeuclidean')
            OT=ot.bregman.sinkhorn_knopp(a[i],b[j],M,reg=reg1)
            W[i][j] = np.trace(np.dot(OT.T,M))
    hot=ot.bregman.sinkhorn_knopp(mu,nu,W,reg=reg2)
    return hot,W


def Mapping(S,T,a,b,HOT):
    index=np.argmax(HOT,1)
    Transported_S=[]
    for i in range(len(S)):
        M=distance.cdist(S[i],T[index[i]], metric='sqeuclidean')
        OT=ot.bregman.sinkhorn_knopp(a[i],b[index[i]],M,reg=0.1)
        Transported_Source=np.linalg.inv(np.diag(OT.dot(np.ones(T[index[i]].shape[0])))).dot(OT).dot(T[index[i]])
        Transported_S=Transported_S+Transported_Source.tolist()
    return Transported_S