# Author: Andrew Davis
# Edited By: Aaron Alphonsus

import h5py
import random

import matplotlib.pyplot as plt
import numpy as np
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
sig2 = 0.1 # data noise

tf = 10.0 # final time
T = 10 # number of observations

# Observation times
tobs = np.linspace(t0, tf, T)
#tobs = sorted(np.random.uniform(t0, tf, T))
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
time = np.linspace(t0, tf, 1000)
xtrue = ForwardModel(time, theta, state0)
fig = MakeFigure(425, 0.9)
ax = plt.gca()
ax.plot(time, xtrue, color='#111111')
ax.plot(tobs, data, 'o', markerfacecolor='#000cff', markeredgecolor='#000cff',
	markersize=8)
#ax.set_title('Harmonic Oscillator True & Noisy Solution', fontsize=16, color='#969696')
ax.set_xlabel('Time t', fontsize=30, color='#969696')
ax.set_ylabel('Position x', fontsize=30, color='#969696')
plt.show()
