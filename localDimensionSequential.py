import numpy as np
from numpy import linalg as LA
import math
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
import scipy as sc
import random
from sklearn.datasets import load_digits, load_iris
from sklearn.neighbors import NearestNeighbors, BallTree
from scipy.stats import norm, chi2
from time import time

#minimum x_bound for when our input is a line
sampleNum = 50
stepSize= 2

#####################################FUNCTIONS###############################
def localDimReduction(data,target,Trials,eta,K, plot,debug):
    #take the data, break it into a tree for k nearest neighbors
    tree = BallTree(data,leaf_size=2)
    #now, we find the K nearest neighbors of each point
    #dis is an array of distances, each row indicates the distance to neighbor
    #ind is the indices of the neighbors
    allLocalDist, ind = tree.query(X,k=K)
    #store a matrix of the local dimensions for each point.
    #the first column is the dimension, the second is t
    localDim = np.zeros((2,len(X[:,0])))
    for i in range(0,len(localDim[0])):
        mu = np.mean(allLocalDist[i])
        sigma = np.std(allLocalDist[i])
        ourT = 2*mu/(sigma**2)
        n = 2*(mu**2)/(sigma**2)
        localDim[0,i]= n
        localDim[1,i] = ourT
    #calculate the global fit n,t
    #print(localDim.shape)
    gDist = calcDist(data)
    useDist = gDist.flatten()
    mu = np.mean(useDist)
    sigma = np.std(useDist)
    globalT = 2*mu/(sigma**2)
    globalN = 2*(mu**2)/(sigma**2)
    #calculate push forward function for global data
    globalDist = ourT*gDist
    newGlobalDist = pushForward(globalDist,globalN)
    #we have the local and global structure, now initialize the low dim data
    #Y = np.random.rand(len(data[:,0]),2)
    Y = np.random.normal(0,0.01,(len(data[:,0]),2))
    trialNum = 0
    while trialNum < Trials:
        #in this for loop, we will only attract
        for i in range(0,len(Y[:,0])):
            #print(i)
            localN = localDim[0,i]
            localT = localDim[1,i]
            for j in range(0,len(Y[:,0])):
                #look through the nearest neighbors
                if j in ind[i] and j != i:
                    itemIndex = np.where(ind[i]==j)
                    k = itemIndex[0][0]
                    dif = Y[i,:]-Y[j,:]
                    dist = LA.norm(dif)
                    dif = (1/dist)*dif
                    actual_y_dist = dist*dist
                    ourDist = allLocalDist[i][k]
                    origDist = localT*ourDist
                    #if dim < 3, just compare distances directly
                    if localN < 3:
                        #too small, pull together
                        if actual_y_dist < origDist:
                            Y[i,:] = Y[i,:]-stepSize*eta*dif
                        #else push apart
                        else:
                            Y[i,:] = Y[i,:] + stepSize*eta*dif
                    #else the dimension is >= 3, use push forward function
                    else:
                        xbound = localN-2
                        newDist = -2*math.log(1-chi2.cdf(origDist,localN))
                        ybound = -2*math.log(1-chi2.cdf(xbound,localN))
                        #if high dim dist is small and low dim dist is large, pull closer
                        if actual_y_dist > newDist and origDist < xbound:
                            Y[i,:] = Y[i,:]-stepSize*eta*dif
                        #if high dim dist is large and low dim dist is small, push apart
                        if actual_y_dist < newDist and origDist > xbound:
                            Y[i,:] = Y[i,:]+stepSize*eta*dif
                #now,let us look at the global points
                if j not in ind[i]:
                    dif = Y[i,:]-Y[j,:]
                    dist = LA.norm(dif)
                    dif = (1/dist)*dif
                    actual_y_dist = dist*dist
                    if globalN < 3:
                        origDist = globalDist[i][j]
                        if actual_y_dist > origDist:
                            Y[i,:] = Y[i,:] - eta*dif
                        else:
                            Y[i,:] = Y[i,:]+eta*dif
                    else:
                        origDist = globalDist[i][j]
                        newDist = newGlobalDist[i][j]
                        xbound = globalN-2
                        ybound = -2*math.log(1-chi2.cdf(xbound,globalN))
                        #if high dim dist is small and low dim dist is large, pull closer
                        if actual_y_dist > newDist and origDist < xbound:
                            Y[i,:] = Y[i,:]-eta*dif
                        if actual_y_dist < newDist and origDist > xbound:
                            Y[i,:] = Y[i,:]+eta*dif
        trialNum += 1
    return(Y,localDim)                                        

#returns a matrix of distnces. B[i,j] is the distance between xi and xj 
def calcDist(x):
    size = len(x[:,1])
    #this will hold the original distances
    B = np.zeros((size,size))
    for i in range(0,size):
        for j in range(i+1,size):
            diff = x[i,:]-x[j,:]
            dists = LA.norm(diff)**2
            B[i,j] = dists
            B[j,i] = B[i,j]
    return(B)

#takes a matrix of distances (x[i][j] is the distance ||xi-xj||)
#and transforms them using the pushforward function
def pushForward(x,n):
    if n < 3:
        return(x)
    size = len(x[:,1])
    dim = n
    #this will hold the transformed distances
    D = np.zeros((size,size))
    for i in range(0,size):
        for j in range(i,size):
            if (1-chi2.cdf(x[i,j],dim))==0:
                D[i,j] = -2*np.log(10**(-292))
            else:
                D[i,j] = -2*math.log(1-chi2.cdf(x[i,j],dim))
            D[j,i] = D[i,j]
    return(D)

#########################RUN DIGITS DATASET#################################
digits = load_digits()
iris = load_iris()
#only running it for 100 values at the moment
X = digits.data[0:50,:]
myTarget = digits.target[0:50]
#K can't be smaller than 32
K = 35
#X = iris.data
#myTarget = iris.target
Y,localDim = localDimReduction(X,myTarget,5,0.01,K, True,False)
plt.scatter(Y[:,0],Y[:,1],c=myTarget)
plt.show()
