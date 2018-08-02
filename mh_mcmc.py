import h5py
import math

import numpy as np
import scipy.stats as stats

from datetime import datetime

from Posterior import Posterior
from MakeFigure import *

# Takes in number of iterations, starting theta point, target distribution and
#     sigma value for the covariance
# Returns arrays with theta1 and theta2 chains along with the acceptance rate
def mh_mcmc(iterations, init_theta, target, prop_sig2, startTime):
    # Define arrays to store the theta1 and theta2 chains. Store the initial
    #     theta values in the first spots of the arrays
    t1 = np.zeros(iterations+1)
    t1[0] = init_theta[0]
    t2 = np.zeros(iterations+1)
    t2[0] = init_theta[1]

    accepted = 0                 # To calculate acceptance rate
    cov = np.diag([prop_sig2]*2) # For the distribution of the proposed point
    theta = init_theta

    for i in range(iterations): 
        # Propose new point
        q_theta = stats.multivariate_normal(theta, cov)
        theta_prop = q_theta.rvs()
        
        # If the proposed point is less than or equal to 0, reject.
        if theta_prop[0] > 0 and theta_prop[1] > 0:
            # Compute acceptance probability
            q_theta_prop = stats.multivariate_normal(theta_prop, cov)
            log_r = (   + target.log_posterior(theta_prop)
                        + q_theta_prop.logpdf(theta)
                        - target.log_posterior(theta)
                        - q_theta.logpdf(theta_prop)
                    )
            log_r = min(0, log_r)

            # Accept/reject point
            u = np.random.uniform()
            if(i % 10 == 0): 
                print(i, datetime.now() - startTime)
            if u < math.exp(log_r):
                theta = theta_prop
                accepted += 1
       
        # Save current point in separate vectors 
        t1[i+1] = theta[0]
        t2[i+1] = theta[1]
    
    return (t1, t2, accepted/iterations)

def WriteData(hdf5file, name, data):
    if name in hdf5file:
        del hdf5file[name]
    hdf5file.create_dataset(name, data=data)

################################# MAIN PROGRAM ################################
# Calls mcmc function and writes generated theta values out to data_mcmc.h5

# Initialize variables
iterations = 10000
init_theta = [1, 0.1]
prop_sig2 = 0.001

# Create Posterior object
a1 = 1 
scale1 = 1 
a2 = 1 
scale2 = 0.1 
hyperparams = [[a1, scale1], [a2, scale2]] 
 
state0 = [308.0, 55.9, 0.5, 0.5]
 
filename = 'iceberg_data.h5'
sig2 = 0.1 
 
target = Posterior(hyperparams, state0, filename, sig2)

# Call mcmc function with initialized variables and posterior object
startTime = datetime.now()
print(startTime)
theta1, theta2, acc_rate = mh_mcmc(iterations, init_theta, target, prop_sig2, 
    startTime)
print(datetime.now(), datetime.now() - startTime)

# Write data out to file
filename = 'iceberg_mcmc.h5'
h5file = h5py.File(filename, 'a')
WriteData(h5file, 'data_mcmc/theta1',   theta1)
WriteData(h5file, 'data_mcmc/theta2',   theta2)
WriteData(h5file, 'data_mcmc/acc_rate', acc_rate)
