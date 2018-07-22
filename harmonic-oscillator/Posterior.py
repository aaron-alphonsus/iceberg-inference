import h5py
import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from ForwardModel import *
from MakeFigure import *

class Posterior:
    def __init__(self, hyperparams, state0, filename, sig2, T):
        self.hyperparams = hyperparams
        self.state0 = state0

        # Read data from file
        hdf5file = h5py.File(filename, 'r')
        self.tobs = hdf5file['data/time'][:]
        xobs = hdf5file['data/xobs'][:]

        # Create covariance matrix
        cov = np.diag([sig2]*T)

        # Create gaussian distribution
        self.like = stats.multivariate_normal(xobs, cov=cov)

    def log_prior(self, theta):
        # Create gamma distributions, find log of the value of the pdf at theta
        hyp = self.hyperparams
        log_g1 = stats.gamma.logpdf(theta[0], a = hyp[0][0], scale = hyp[0][1])
        log_g2 = stats.gamma.logpdf(theta[1], a = hyp[1][0], scale = hyp[1][1])
        
        return log_g1 + log_g2 

    def log_likelihood(self, theta):     
        # Call forward model, store the true position, and evaluate the log of 
        # the pdf at xtrue
        xtrue = ForwardModel(self.tobs, theta, self.state0)
        likelihood = self.like.logpdf(xtrue)
        
        return likelihood

    def log_density(self, theta):
        return self.log_prior(theta) + self.log_likelihood(theta)

################################# Main Program ################################

if __name__ == '__main__':
    
    # Construct posterior object
    a1 = 2
    scale1 = 1
    a2 = 1
    scale2 = 1
    hyperparams = [[a1, scale1], [a2, scale2]]
    
    x0 = 1.0
    u0 = 0.0
    state0 = [x0, u0]
    
    filename = "data.h5"
    
    sig2 = 0.01
    T = 10
    
    post = Posterior(hyperparams, state0, filename, sig2, T)
    
    # Create linearly spaced theta values to evaluate posterior at
    t1_size = 100
    t2_size = 120
    theta1 = np.linspace(0.1, 8, t1_size)
    theta2 = np.linspace(0, 4, t2_size)
    
    # Fill prior, likelihood, and posterior matrices for plotting purposes
    prior = np.empty(shape = (t2_size, t1_size))
    likelihood = np.empty(shape = (t2_size, t1_size))
    posterior = np.empty(shape = (t2_size, t1_size))
    for i in range(posterior.shape[0]): 
        for j in range(posterior.shape[1]): 
            prior[i][j] = post.log_prior([theta1[j], theta2[i]])
            likelihood[i][j] = post.log_likelihood([theta1[j], theta2[i]])
            posterior[i][j] = prior[i][j] + likelihood[i][j]
    
    # Plot contour graphs for prior, likelihood, and posterior
    fig = MakeFigure(700, 0.75)
    ax = plt.gca()
    ax.contourf(theta1, theta2, prior)
    ax.set_title('Prior', fontsize = 16)
    ax.set_xlabel('Spring Coefficient ($k/m$)', fontsize = 12)
    ax.set_ylabel('Damping Coefficient ($c/m$)', fontsize = 12)
    
    fig = MakeFigure(700, 1)
    ax = plt.gca()
    ax.contourf(theta1, theta2, likelihood)
    ax.set_title('Likelihood', fontsize = 16)
    ax.set_xlabel('Spring Coefficient ($k/m$)', fontsize = 12)
    ax.set_ylabel('Damping Coefficient ($c/m$)', fontsize = 12)
    
    fig = MakeFigure(700, 1)
    ax = plt.gca()
    ax.contourf(theta1, theta2, posterior)
    ax.set_title('Posterior', fontsize = 16)
    ax.set_xlabel('Spring Coefficient ($k/m$)', fontsize = 12)
    ax.set_ylabel('Damping Coefficient ($c/m$)', fontsize = 12)
    
    plt.show()
