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
#from categorical_scatter import categorical_scatter_2d
###source code: https://nlml.github.io/in-raw-numpy/in-raw-numpy-t-sne/##

#calculates the negative squared euclidean dist for all pairs point
def neg_squared_euc_dists(X):
    sum_X = np.sum(np.square(X),1)
    D = np.add(np.add(-2*np.dot(X,X.T),sum_X).T,sum_X)
    return(-D)

#takes exp/sum exp for each row of X
def softmax(X, diag_zero=True):
    # subtract max for numerical stability
    e_x = np.exp(X-np.max(X,axis=1).reshape([-1,1]))

    #we want diagonal prob to be 0
    if diag_zero:
        np.fill_diagonal(e_x,0.)

    #add a tiny constant for stability of log
    e_x = e_x + 1e-8

    return(e_x/e_x.sum(axis=1).reshape([-1,1]))

#calc the probability matrix
def calc_prob_matrix(distances,sigmas=None):
    if sigmas is not None:
        two_sig_sq = 2. * np.square(sigmas.reshape((-1,1)))
        return(softmax(distances/two_sig_sq))
    else:
        return(softmax(distances))

#binary search function we will use the find the sigmas
def binary_search(eval_fn, target, tol=1e-10, max_iter=10000, lower=1e-20, upper=1000.):
    for i in range(max_iter):
        guess = (lower + upper)/2.
        val = eval_fn(guess)
        if val > target:
            upper = guess
        else:
            lower = guess
        if np.abs(val-target) <= tol:
            break
    return guess

def calc_perplexity(prob_matrix):
    entropy = -np.sum(prob_matrix * np.log2(prob_matrix), 1)
    perplexity = 2 ** entropy
    return perplexity

##calculate the perplexity for all the distances and the sigmas
def perplexity(distances, sigmas):
    return calc_perplexity(calc_prob_matrix(distances, sigmas))

###give it the distances, find the sigma###
def find_optimal_sigmas(distances, target_perplexity):
    sigmas = [] 
    # For each row of the matrix (each point in our dataset)
    for i in range(distances.shape[0]):
        # Make fn that returns perplexity of this row given sigma
        eval_fn = lambda sigma: \
            perplexity(distances[i:i+1, :], np.array(sigma))
        # Binary search over sigmas to achieve target perplexity
        correct_sigma = binary_search(eval_fn, target_perplexity)
        # Append the resulting sigma to our output array
        sigmas.append(correct_sigma)
    return np.array(sigmas)

###q joint distributions (not t)
def q_joint(Y):
    # Get the distances from every point to every other
    distances = neg_squared_euc_dists(Y)
    # Take the elementwise exponent
    exp_distances = np.exp(distances)
    # Fill diagonal with zeroes so q_ii = 0
    np.fill_diagonal(exp_distances, 0.)
    # Divide by the sum of the entire exponentiated matrix
    return exp_distances / np.sum(exp_distances), None


###p conditional distributions
def p_conditional_to_joint(P):
    return (P + P.T) / (2. * P.shape[0])

###p joint distributions
def p_joint(X, target_perplexity):
    # Get the negative euclidian distances matrix for our data
    distances = neg_squared_euc_dists(X)
    # Find optimal sigma for each row of this distances matrix
    sigmas = find_optimal_sigmas(distances, target_perplexity)
    # Calculate the probabilities based on these optimal sigmas
    p_conditional = calc_prob_matrix(distances, sigmas)
    # Go from conditional to joint probabilities matrix
    P = p_conditional_to_joint(p_conditional)
    return P

##symmetric SNE Grad
def symmetric_sne_grad(P, Q, Y, _):
    pq_diff = P - Q  # NxN matrix
    pq_expanded = np.expand_dims(pq_diff, 2)  #NxNx1
    y_diffs = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)  #NxNx2
    grad = 4. * (pq_expanded * y_diffs).sum(1)  #Nx2
    return grad

##estimate SNE/TSNE
def estimate_sne(X, y, P, rng, num_iters, q_fn, grad_fn, learning_rate,
                 momentum, plot):
    """Estimates a SNE model.

    # Arguments
        X: Input data matrix.
        y: Class labels for that matrix.
        P: Matrix of joint probabilities.
        rng: np.random.RandomState().
        num_iters: Iterations to train for.
        q_fn: Function that takes Y and gives Q prob matrix.
        plot: How many times to plot during training.
    # Returns:
        Y: Matrix, low-dimensional representation of X.
    """

    # Initialise our 2D representation
    Y = rng.normal(0., 0.0001, [X.shape[0], 2])

    # Initialise past values (used for momentum)
    if momentum:
        Y_m2 = Y.copy()
        Y_m1 = Y.copy()

    # Start gradient descent loop
    for i in range(num_iters):

        # Get Q and distances (distances only used for t-SNE)
        Q, distances = q_fn(Y)
        # Estimate gradients with respect to Y
        grads = grad_fn(P, Q, Y, distances)

        # Update Y
        Y = Y - learning_rate * grads
        if momentum:  # Add momentum
            Y += momentum * (Y_m1 - Y_m2)
            # Update previous Y's for momentum
            Y_m2 = Y_m1.copy()
            Y_m1 = Y.copy()

        # Plot sometimes
        #if plot and i % (num_iters / plot) == 0:
            #print(Y.shape)
            #plt.scatter(Y[:,0],Y[:,1],c=y)
            #categorical_scatter_2d(Y, y, alpha=1.0, ms=6,
                                   #show=True, figsize=(9, 6))

    return Y

###let's assume q is according to t-dist
def q_tsne(Y):
    distances = neg_squared_euc_dists(Y)
    inv_distances = np.power(1. - distances, -1)
    np.fill_diagonal(inv_distances, 0.)
    return inv_distances / np.sum(inv_distances), inv_distances


###calc the tsne gradient
def tsne_grad(P, Q, Y, inv_distances):
    """Estimate the gradient of t-SNE cost with respect to Y."""
    pq_diff = P - Q
    pq_expanded = np.expand_dims(pq_diff, 2)
    y_diffs = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)

    # Expand our inv_distances matrix so can multiply by y_diffs
    distances_expanded = np.expand_dims(inv_distances, 2)

    # Multiply this by inverse distances matrix
    y_diffs_wt = y_diffs * distances_expanded

    # Multiply then sum over j's
    grad = 4. * (pq_expanded * y_diffs_wt).sum(1)
    return grad

# Set global parameters
NUM_POINTS = 200            # Number of samples from MNIST
#CLASSES_TO_USE = [0, 1, 8]  # MNIST classes to use
PERPLEXITY = 20
SEED = 1                    # Random seed
MOMENTUM = 0.9
LEARNING_RATE = 10.
NUM_ITERS = 500             # Num iterations to train for
TSNE = True                # If False, Symmetric SNE
NUM_PLOTS = 5               # Num. times to plot in training


def main():
    # numpy RandomState for reproducibility
    rng = np.random.RandomState(SEED)

##    # Load the first NUM_POINTS 0's, 1's and 8's from MNIST
##    X, y = load_mnist('datasets/',
##                      digits_to_keep=CLASSES_TO_USE,
##                      N=NUM_POINTS)
    digits = load_digits()
    X = digits.data
    y = digits.target

    # Obtain matrix of joint probabilities p_ij
    P = p_joint(X, PERPLEXITY)

    # Fit SNE or t-SNE
    Y = estimate_sne(X, y, P, rng,
             num_iters=NUM_ITERS,
             q_fn=q_tsne if TSNE else q_joint,
             grad_fn=tsne_grad if TSNE else symmetric_sne_grad,
             learning_rate=LEARNING_RATE,
             momentum=MOMENTUM,
             plot=NUM_PLOTS)
    plt.scatter(Y[:,0],Y[:,1],c=y)
    plt.show()
    
