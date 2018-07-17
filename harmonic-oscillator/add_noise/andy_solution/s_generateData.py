# Author: Andrew Davis

import numpy as np

import h5py

import matplotlib.pyplot as plt
from matplotlib import rcParams

from math import *

from ForwardModel import *

# hdf5file - writeable hdf5 file
# name - name of the dataset
# data - data to write to file
def WriteData(hdf5file, name, data):
    if name in hdf5file:
        del hdf5file[name]
    # print(tobs)
    # print(data)
    hdf5file.create_dataset(name, data=data)

# put this in a callable script
def MakeFigure(totalWidthPts, fraction, presentationVersion = False):
    fig_width_pt  = totalWidthPts * fraction
    
    inches_per_pt = 1.0 / 72.27
    golden_ratio  = (np.sqrt(5) - 1.0) / 2.0  # because it looks good
    
    fig_width_in  = fig_width_pt * inches_per_pt  # figure width in inches
    fig_height_in = fig_width_in * golden_ratio   # figure height in inches
    fig_dims      = [fig_width_in, fig_height_in] # fig dims as a list
    fig = plt.figure(figsize = fig_dims)
    
    greyColor = '#969696'
    whiteColor = '#ffffff'
    if not presentationVersion:
        rcParams['axes.labelsize'] = 9
        rcParams['xtick.labelsize'] = 9
        rcParams['ytick.labelsize'] = 9
        rcParams['legend.fontsize'] = 9
    else:
        rcParams['axes.labelsize'] = 12
        rcParams['xtick.labelsize'] = 12
        rcParams['ytick.labelsize'] = 12
        rcParams['legend.fontsize'] = 12
    rcParams['axes.edgecolor'] = greyColor
    rcParams['axes.facecolor'] = whiteColor
    rcParams['figure.facecolor'] = whiteColor
    rcParams['axes.labelcolor'] = greyColor
    rcParams['text.color'] = greyColor
    rcParams['xtick.color'] = greyColor
    rcParams['ytick.color'] = greyColor
    
    rcParams['lines.antialiased'] = True
    if not presentationVersion:
        rcParams['font.family'] = 'sans-serif'
        rcParams['font.serif'] = ['Computer Modern Roman']
        rcParams['text.usetex'] = True
    else:
        rcParams['text.usetex'] = False          
        rcParams['font.family'] = 'sans-serif'
        rcParams['lines.linewidth'] = 1.5
        
    return fig


################################ MAIN PROGRAM

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
ax.plot(tobs, data, 'o', markerfacecolor='#969696', markeredgecolor='#969696', markersize=4)
ax.set_xlabel('Time $t$', fontsize=16, color='#969696')
ax.set_ylabel('Position $x$', fontsize=16, color='#969696')
plt.savefig('fig_data.pdf', format='pdf', bbox_inches='tight')
