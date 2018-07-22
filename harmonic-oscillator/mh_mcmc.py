import math

import numpy as np
import scipy.stats as stats

from Posterior import Posterior
from MakeFigure import *

def mh_mcmc(iterations, theta, t1, t2, stationary, sigma): 
    cov = np.diag([sigma]*2)
    for i in range(iterations): 
        
        # Propose new point
        q_theta = stats.multivariate_normal(theta, cov)
        theta_prop = q_theta.rvs()
        # This is a hack. TODO: Find an alternative 
        while theta_prop[0] <= 0 or theta_prop[1] <= 0:
            theta_prop = q_theta.rvs()         
 
        # Compute acceptance probability
        q_theta_prop = stats.multivariate_normal(theta_prop, cov)
        log_r = (   + stationary.log_density(theta_prop)  
                    + q_theta_prop.logpdf(theta) 
                    - stationary.log_density(theta) 
                    - q_theta.logpdf(theta_prop)
                )
        alpha = min(1, math.exp(log_r))
        
        # Accept/reject point
        u = np.random.uniform()
        if u < alpha:
            theta = theta_prop
       
        # Save current point in separate vectors 
        t1[i+1] = theta[0]
        t2[i+1] = theta[1]
    
    # Plot mixing graphs
    fig = MakeFigure(450, 1)
    ax = plt.gca()
    ax.set_title('Theta1 Mixing', fontsize = 12)
    ax.set_xlabel('Iterations', fontsize = 10)
    ax.set_ylabel('Theta1', fontsize = 10)
    ax.plot(t1)

    fig = MakeFigure(450, 1)
    ax = plt.gca() 
    ax.set_title('Theta2 Mixing', fontsize = 12)
    ax.set_xlabel('Iterations', fontsize = 10)
    ax.set_ylabel('Theta2', fontsize = 10)
    ax.plot(t2)

    fig = MakeFigure(450, 1)
    ax = plt.gca() 
    ax.set_title('Scatter Plot', fontsize = 12)
    ax.set_xlabel('Theta1', fontsize = 10)
    ax.set_ylabel('Theta2', fontsize = 10)
    ax.scatter(t1, t2)

    fig = MakeFigure(450, 1)
    ax = plt.gca()
    ax.set_title('Histogram', fontsize = 12)
    ax.set_xlabel('Theta1', fontsize = 10)
    ax.set_ylabel('Theta2', fontsize = 10)
    hist = ax.hist2d(t1, t2, bins=(50,50), cmap=plt.cm.inferno)
    plt.colorbar(hist[3], ax=ax)
    plt.show()

# Initialize variables
iterations = 100000
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
 
stationary = Posterior(hyperparams, state0, filename, sig2, T)

# Call mcmc function with initialized variables and posterior object
mh_mcmc(iterations, init_theta, t1, t2, stationary, sigma)

