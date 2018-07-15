import h5py
import math
import numpy as np
import scipy.stats as stats

from matplotlib import pyplot as plt

from ForwardModel import *

def likelihood(filename, T, sig2, theta, state0):
    
    # Read data from file
    hdf5file = h5py.File(filename, 'r')
    # hdf5file.visit(print)
    tobs = hdf5file['data/time'][:]
    xobs = hdf5file['data/xobs'][:]   

    # Create covariance matrix
    cov = np.diag([sig2]*T)

    # Create gaussian distribution
    like = stats.multivariate_normal(xobs, cov) 
    

    return like.pdf(ForwardModel(tobs, theta, state0))

def posterior(theta, state0):
    
    # log_prior = log(stats.gamma.pdf(t1, a = 2, scale = 1)) + log(gamma(t2, a = 2, scale = 0.25))
     
    # x = np.linspace(0, 100, 200)
    # y = stats.gamma.pdf(x, a = 29, scale = 0.3333)
    # plt.plot(x, y)
    # plt.show()

    T = 10
    filename = "data.h5"
    sig2 = 0.01
    log_like = math.log(likelihood(filename, T, sig2, theta, state0)) 
    # print(log_like)

    # return log_prior + log_like

k = 1.0
c = 0.25
m = 1.0
theta = [math.sqrt(k/m), c/m]
x0 = 1.0
u0 = 0.0
state0 = [x0, u0]

posterior(theta, state0)

# build structure to pass in args
# pass in a and scale
# create theta1 and theta2 vectors (0.1, 20, 1000) and (0.0, 30, 1500) (recheck your a and scale)
# take log before returning
