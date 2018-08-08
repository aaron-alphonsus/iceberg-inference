# Authors: Chi Zhang, Aaron Alphonsus
###############################################################################

import h5py

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from Forward import *
from MakeFigure import *

class Posterior:
    def __init__(self, hyperparams, state0, filename, sig2):
        self.hyperparams = hyperparams
        self.state0 = state0

        # Read data from file
        hdf5file = h5py.File(filename, 'r')
        self.tobs = hdf5file['data/time'][:]
        xobs = hdf5file['data/xobs'][:]
        yobs = hdf5file['data/yobs'][:]

        # Create covariance matrix
        cov = np.diag([sig2]*xobs.size)

        # Create gaussian distributions for x and y
        self.x_like = stats.multivariate_normal(xobs, cov=cov)
        self.y_like = stats.multivariate_normal(yobs, cov=cov)

        [self.values, self.values2] = fixSecondMess()

    def log_prior(self, theta):
        # Create gamma distributions, find log of the pdf at theta
        hyp = self.hyperparams
        log_g1 = stats.gamma.logpdf(theta[0], a = hyp[0][0], scale = hyp[0][1])
        log_g2 = stats.gamma.logpdf(theta[1], a = hyp[1][0], scale = hyp[1][1])

        return log_g1 + log_g2

    def log_likelihood(self, theta):
        # Call forward model, store the true position, evaluate log of the pdf
        # at xtrue
        ObsData = ForwardModel(self.tobs, theta, self.state0, self.values,
            self.values2)
        xtrue = [x[0] for x in ObsData]
        ytrue = [y[1] for y in ObsData]
        log_x_like = self.x_like.logpdf(xtrue)
        log_y_like = self.y_like.logpdf(ytrue)
        return log_x_like + log_y_like

    def log_posterior(self, theta):
        log_posterior = self.log_prior(theta) + self.log_likelihood(theta)
        return log_posterior

def WriteData(hdf5file, name, data):
    if name in hdf5file:
        del hdf5file[name]
    hdf5file.create_dataset(name, data=data)
################################# Main Program ################################

if __name__ == '__main__':
    filename = "iceberg_data.h5"
    sig2 = 0.01

    a1 = 2
    scale1 = 1
    a2 = 1
    scale2 = 1
    hyperparameters = [[a1, scale1], [a2, scale2]]

    theta = [1, 0.1]
    state0 = [308.0, 55.9, 0.5, 0.5]

    post = Posterior(hyperparameters, state0, filename, sig2)

    # Create linearly spaced theta values to evaluate posterior at
    t1_size = 10
    t2_size = 10
    theta1 = np.linspace(0.01, 1.5, t1_size)
    theta2 = np.linspace(0.01, 1, t2_size)

    log_prior = np.empty(shape = (t2_size, t1_size))
    log_likelihood = np.empty(shape = (t2_size, t1_size))
    log_posterior = np.empty(shape = (t2_size, t1_size))
    for i in range(t2_size):
        for j in range(t1_size):
            log_prior[i][j] = post.log_prior([theta1[j], theta2[i]])
            log_likelihood[i][j] = post.log_likelihood([theta1[j], theta2[i]])
            log_posterior[i][j] = log_prior[i][j] + log_likelihood[i][j]
        print(i)

    filename2 = 'iceberg_posterior_linear_spaced.h5'
    h5file3 = h5py.File(filename2, 'a')
    WriteData(h5file3, 'data/theta1',   theta1)
    WriteData(h5file3, 'data/theta2',   theta2)
    WriteData(h5file3, 'data/prior',   np.exp(np.array(log_prior)))
    WriteData(h5file3, 'data/likelihood',   np.exp(np.array(log_likelihood)))
    WriteData(h5file3, 'data/posterior',   np.exp(np.array(log_posterior)))


    fig = MakeFigure(700, 0.75)
    ax = plt.gca()
    ax.contourf(theta1, theta2, np.exp(np.array(log_prior)))
    ax.set_title('Prior', fontsize = 16)
    ax.set_xlabel('Water Coefficient ($Cw$)', fontsize = 12)
    ax.set_ylabel('Air Coefficient ($Ca$)', fontsize = 12)

    fig = MakeFigure(700, 1)
    ax = plt.gca()
    cont = ax.contourf(theta1, theta2, np.exp(np.array(log_likelihood)))
    plt.colorbar(cont)
    ax.set_title('Likelihood', fontsize = 16)
    ax.set_xlabel('Water Coefficient ($Cw$)', fontsize = 12)
    ax.set_ylabel('Air Coefficient ($Ca$)', fontsize = 12)

    fig = MakeFigure(700, 1)
    ax = plt.gca()
    cont = ax.contourf(theta1, theta2, np.exp(np.array(log_posterior)))
    plt.colorbar(cont)
    ax.set_title('Posterior', fontsize = 16)
    ax.set_xlabel('Water Coefficient ($Cw$)', fontsize = 12)
    ax.set_ylabel('Air Coefficient ($Ca$)', fontsize = 12)

    plt.show()
