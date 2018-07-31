import h5py

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from IcebergForwardModel_Sketch import *
from harmonic_oscillator.MakeFigure import *

class Posterior:
    def __init__(self, hyperparams, state0, filename, sig2):
        self.hyperparams = hyperparams
        self.state0 = state0

        hdf5file = h5py.File(filename, 'r')
        self.tobs = hdf5file['data/time'][:]
        xobs = hdf5file['data/xobs'][:] 
        yobs = hdf5file['data/yobs'][:]

        cov = np.diag([sig2]*xobs.size)

        self.x_like = stats.multivariate_normal(xobs, cov=cov)
        self.y_like = stats.multivariate_normal(yobs, cov=cov)

    def log_prior(self, theta):    
        hyp = self.hyperparams
        log_g1 = stats.gamma.logpdf(theta[0], a = hyp[0][0], scale = hyp[0][1]) 
        log_g2 = stats.gamma.logpdf(theta[1], a = hyp[1][0], scale = hyp[1][1])
    
        return log_g1 + log_g2
    
    def log_likelihood(self, theta):    
        ObsData = ForwardModel(self.tobs, theta, self.state0)
        xtrue = [x[0] for x in ObsData]
        ytrue = [y[1] for y in ObsData]

        log_x_like = self.x_like.logpdf(xtrue)        
        log_y_like = self.y_like.logpdf(ytrue)
        return log_x_like + log_y_like

filename = "iceberg_data.h5"
sig2 = 0.01

a1 = 2
scale1 = 1
a2 = 1
scale2 = 1
hyperparameters = [[a1, scale1], [a2, scale2]]

# theta = [1, 0.001]
# x0, y0, u0, v0 = 320.0, 46.666666, -0.371430489, 0.123941015917
# state0 = [x0, y0, u0, v0]
theta = [1.5, 1.5]
state0 = [0, -1, 0, 0]

post = Posterior(hyperparameters, state0, filename, sig2)

# Create linearly spaced theta values to evaluate posterior at
t1_size = 8
t2_size = 8
theta1 = np.linspace(0.1, 5, t1_size)
theta2 = np.linspace(0, 5, t2_size)

prior = np.empty(shape = (t2_size, t1_size))
likelihood = np.empty(shape = (t2_size, t1_size))
# posterior = np.empty(shape = (t2_size, t1_size))
for i in range(t2_size): 
    for j in range(t1_size): 
        prior[i][j] = post.log_prior([theta1[j], theta2[i]])
        likelihood[i][j] = post.log_likelihood([theta1[j], theta2[i]])
        # posterior[i][j] = prior[i][j] + likelihood[i][j]

fig = MakeFigure(700, 0.75)
ax = plt.gca()
ax.contourf(theta1, theta2, prior)
ax.set_title('Prior', fontsize = 16)
ax.set_xlabel('Water Coefficient ($Cw$)', fontsize = 12)
ax.set_ylabel('Air Coefficient ($Ca$)', fontsize = 12)

fig = MakeFigure(700, 1)
ax = plt.gca()
ax.contourf(theta1, theta2, likelihood)
ax.set_title('Likelihood', fontsize = 16)
ax.set_xlabel('Water Coefficient ($Cw$)', fontsize = 12)
ax.set_ylabel('Air Coefficient ($Ca$)', fontsize = 12)

plt.show()
