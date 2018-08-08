# Authors: Chi Zhang, Aaron Alphonsus
###############################################################################
import h5py
import numpy as np

from IcebergForwardModel_Sketch import ForwardModel
from MakeFigure import *

def WriteData(hdf5file, name, data):
    if name in hdf5file:
        del hdf5file[name]
    hdf5file.create_dataset(name, data=data)

filename = 'iceberg_data.h5'
hdf5file = h5py.File(filename, 'a')

t0 = 0.0
sig2 = 0.1

theta = [1.5, 1.5]
state0 = [0,-1,0,0]
# theta = [1, 0.001]
# x0, y0, u0, v0 = 320.0, 46.666666, -0.371430489, 0.123941015917
# state0 = [x0, y0, u0, v0]

tf = 100
# tf = 36
T = 200

tobs = np.linspace(t0, tf, T)
WriteData(hdf5file, 'data/time', tobs)

# TODO: Don't copy x and y values into new arrays
ObsData = ForwardModel(tobs, theta, state0)
xobs = [x[0] for x in ObsData]
yobs = [y[1] for y in ObsData]

cov = np.diag([sig2]*T)
x_noise = np.random.multivariate_normal([0.0]*T, cov)
y_noise = np.random.multivariate_normal([0.0]*T, cov)

x_data = xobs + x_noise
y_data = yobs + y_noise
WriteData(hdf5file, 'data/xobs', x_data)
WriteData(hdf5file, 'data/yobs', y_data)

time = np.linspace(t0, tf, 1000)
TrueData = ForwardModel(time, theta, state0)
xtrue = [x[0] for x in TrueData]
ytrue = [y[1] for y in TrueData]

fig = MakeFigure(425, 0.9)
ax = plt.gca()
ax.plot(xtrue, ytrue, color = '#111111')
ax.plot(x_data, y_data, 'o', markerfacecolor = '#000cff',
    markeredgecolor = '#000cff', markersize = 8)
ax.set_xlabel('Longitude', fontsize = 30, color = '#969696')
ax.set_ylabel('Latitude', fontsize = 30, color = '#969696')
plt.show()
