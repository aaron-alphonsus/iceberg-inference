# Author: Andrew Davis
# Edited By: Aaron Alphonsus

import numpy as np

import h5py

import matplotlib.pyplot as plt
from matplotlib import rcParams

from math import *

from ForwardModel import *
from MakeFigure import *

# hdf5file - writeable hdf5 file
# name - name of the dataset
# data - data to write to file
def WriteData(hdf5file, name, data):
    if name in hdf5file:
        del hdf5file[name]
    # print(tobs)
    # print(data)
    hdf5file.create_dataset(name, data=data)

################################ MAIN PROGRAM ################################

# data file
filename = 'data.h5'
hdf5file = h5py.File(filename, 'a')

# initial condition
x0 = 1.0
u0 = 0.0
state0 = [x0, u0] # initial state
t0 = 0.0 # initial time

# constants
k = 1.0
c = 0.25
m = 1.0
theta = [sqrt(k/m), c/m]

# hyperparameters
sig2 = 0.01 # data noise

tf = 10.0 # final time
T = 10 # number of observations

# observations times
tobs = np.linspace(0.0, tf, T)
WriteData(hdf5file, 'data/time', tobs)

# run the forward model
xobs = ForwardModel(tobs, theta, state0)

# generate the noise 
cov = np.diag([sig2]*T)
noise = np.random.multivariate_normal([0.0]*T, cov)

# add noise to observations
data = xobs+noise
WriteData(hdf5file, 'data/xobs', data)

# for plotting purposes, compute the truth
time = np.linspace(0.0, tf, 1000)
xtrue = ForwardModel(time, theta, state0)

fig = MakeFigure(425, 0.9)
ax = plt.gca()
ax.plot(time, xtrue, color='#969696')
ax.plot(tobs, data, 'o', markerfacecolor='#969696', markeredgecolor='#969696',
	markersize=4)
ax.set_xlabel('Time $t$', fontsize=16, color='#969696')
ax.set_ylabel('Position $x$', fontsize=16, color='#969696')
plt.savefig('fig_data.pdf', format='pdf', bbox_inches='tight')
