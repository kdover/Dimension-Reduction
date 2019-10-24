import numpy as np
from numpy import linalg as LA
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
import scipy as sc
import random
from sklearn.datasets import load_digits, load_iris
from scipy.stats import norm, chi2
from time import time

#minimum x_bound for when our input is a line
min_x_bound = 100

#####################################FUNCTIONS###############################
#simple gradient: -1, 0, 1 depending on distances x and y, use transformed distances
#data is your data (each row is an element)
#target is the target values for classifying (just for plotting)
#Trials is the number of times you want to run the algorithm
#eta is our step size
#stopVal is how small we are willing to let the distances get
#plot is boolean. If true, plot the values, if not, don't plot
#debug is a boolean that allows me to print stuff while debugging
#function returns a 2xN matrix Y (low dimensional representation)
def gradChiKSquare(data,target,Trials,eta, plot,debug):
    t0 = time()
    #calculate the dimension and t of this data
    origDist1 = calcDist(data)
    ourData = origDist1.flatten()
    mu = np.mean(ourData)
    std = np.std(ourData)
    t = 2*mu/(std**2)
    n = 2*(mu**2)/(std**2)
    #what dimension are we working in 
    dim = n
    #if n < 1, make it 1
    if n < 1:
        dim =1
    origDist = t*origDist1
    #now get the transformed distances of the shifted, scaled dataset
    newDist = pushForward(origDist,dim)
    #given the x bound, what is the y bound?
    x_bound = dim-2
    #if dimension is 1, the relationship is linear
    if x_bound <= 0 or n < 2 or debug == True:
        x_bound = min_x_bound
        y_bound = x_bound
    else:
        y_bound = -2*math.log(1-chi2.cdf(x_bound,dim))
    #initialize some random data, make a copy to store things during our calculations
    Y = np.random.rand(2,len(data[:,0]))
    t = 0
    #averages = []
    while t < Trials:
        for i in range(0,len(Y[0,:])):
            for j in range(0,len(Y[1,:])):
                if(i != j):
                    #calc the normalized difference
                    #calc the distance
                    dif = Y[:,i]-Y[:,j]
                    dist = LA.norm(dif)
                    dif = (1/dist)*dif
                    actual_y_dist = dist*dist
                    #if our high dimensional distance is small and the low dimensional distance is large, pull closer together
                    if actual_y_dist > newDist[i][j] and origDist[i][j] < x_bound:
                        Y[:,i] = Y[:,i] - eta*dif
                    #if our high dimension distance is large and our low dimensional distance is small, push apart.
                    if actual_y_dist < y_bound and origDist[i][j] > x_bound:
                        Y[:,i] = Y[:,i] + eta*dif
        t += 1
    t1 = time()
    if plot == True:
        (fig,subplots) = plt.subplots(1,2,figsize=(15,8))
        ax = subplots[0]
        ax.hist(origDist.flatten(),density=True)
        xx = np.linspace(0,max(origDist.flatten()),1000)
        yy = chi2.pdf(xx,dim)
        ax.plot(xx,yy,c='r')
        ax = subplots[1]
        ax.scatter(Y[0,:],Y[1,:],c=target)
        plt.show()
    print('new dim '+str(dim))
    print('Time '+str(t1-t0))
    return(Y)
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
    if n < 2:
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
#only running it for 100 values at the moment
X = digits.data[0:100]
myTarget = digits.target[0:100]
Y = gradChiKSquare(X,myTarget,5,0.01,True,False)
plt.show()
