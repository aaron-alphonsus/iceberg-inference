import h5py
import math

import numpy as np
import scipy.stats as stats

from ForwardModel import *

class Posterior:
    
    def __init__(self, hyperparams, state0, filename, sig2, T):
        self.hyperparams = hyperparams
        self.state0 = state0
        self.filename = filename
        self.sig2 = sig2
        self.T = T

    def log_prior(self, theta):
        hyp = self.hyperparams
        gamma1 = stats.gamma.pdf(theta[0], a = hyp[0][0], scale = hyp[0][1])
        gamma2 = stats.gamma.pdf(theta[1], a = hyp[1][0], scale = hyp[1][1])
        return math.log(gamma1) + math.log(gamma2) 

    def log_likelihood(self, theta):
        
        # Read data from file
        hdf5file = h5py.File(self.filename, 'r')
        # hdf5file.visit(print)
        tobs = hdf5file['data/time'][:]
        xobs = hdf5file['data/xobs'][:]

        # Create covariance matrix
        cov = np.diag([self.sig2]*self.T)

        # Create gaussian distribution
        like = stats.multivariate_normal(xobs, cov)

        # Call forward model, store the true position and evaluate the likelihood
        # at xtrue
        xtrue = ForwardModel(tobs, theta, self.state0)
        likelihood = like.pdf(xtrue)
        
        return math.log(likelihood)

    def log_density(self, theta):
        return self.log_prior(theta) + self.log_likelihood(theta)

################################# Main Program #################################

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

theta1 = np.linspace(0.1, 20, 1000)
theta2 = np.linspace(0.0, 30, 1500)
for t1 in theta1:
    for t2 in theta2:
        print(post.log_density([t1, t2]))
