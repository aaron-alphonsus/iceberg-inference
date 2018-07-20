import math

import numpy as np
import scipy.stats as stats

from Posterior import Posterior

def mh_mcmc(iterations, theta, t1, t2, target, sigma): 
    for i in range(iterations): 
        # Propose new point
        cov = np.diag([sigma]*2)
        q_theta = stats.multivariate_normal(theta, cov)
        theta_prop = theta + q_theta.rvs()
        print("prop = ", theta_prop)
        while theta_prop[0] <= 0 or theta_prop[1] <= 0:
            theta_prop = theta + q_theta.rvs() 
            print("prop = ", theta_prop)         
 
        # Compute acceptance probability
        q_theta_prop = stats.multivariate_normal(theta_prop, cov)
        log_r = (target.log_density(theta) 
                    + q_theta_prop.logpdf(theta) 
                    - target.log_density(theta_prop) 
                    - q_theta.logpdf(theta_prop)
                )
        alpha = min(1, math.exp(log_r))
        
        # Accept/reject point
        u = np.random.uniform()
        if u < alpha:
            theta = theta_prop
        
        print("t = ", theta)
        t1[i+1] = theta[0]
        t2[i+1] = theta[1]
    
    print(t1)
    print(t2)

# Initialize variables

iterations = 500
init_theta = [1, 0.25]
sigma = 0.01

t1 = np.zeros(iterations+1)
t1[0] = init_theta[0]
t2 = np.zeros(iterations+1)
t2[0] = init_theta[1]

# Create Posterior object

a1 = 2 
scale1 = 1 
a2 = 1 
scale2 = 1 
hyperparams = [[a1, scale1], [a2, scale2]] 
 
x0 = 1.0 
u0 = 0.0 
state0 = [x0, u0] 
 
filename = "data.h5" 
 
sig2 = 0.1 
T = 10 
 
target = Posterior(hyperparams, state0, filename, sig2, T)

# Call mcmc function
mh_mcmc(iterations, init_theta, t1, t2, target, sigma)

