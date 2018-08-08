#---------------modules--------------------
import numpy as np
import math as math
from scipy import stats
import h5py
import matplotlib.pyplot as plt
from matplotlib import rcParams
from math import *
from MakeFigure import *
from ForwardModel import *
from scipy.integrate import ode
#----------read data--------------------------------
filename = 'data_mcmc.h5'
hdf5file = h5py.File(filename, 'r')
theta_1 = hdf5file['data_mcmc/theta1'].value
theta_2 = hdf5file['data_mcmc/theta2'].value
#-------------------main code------------------
def WriteData(hdf5file, name, data):
    if name in hdf5file:
        del hdf5file[name]
    hdf5file.create_dataset(name, data=data)
state0 = [1, 0] # inital state
# #timeArray = list(np.linspace(0, 10, 10))
# #T = len(timeArray)
# positionArray = np.array([[0.0]*200]*(len(theta_1)))
# timeMat = np.array([[0.0]*200]*(len(theta_1)))
# for i in range(0, len(theta_1)):
#     timeArray = sorted(np.random.uniform(0, 10, 200))
#     positionArray[i,:] = ForwardModel(timeArray, [theta_1[i], theta_2[i]], state0)
#     timeMat[i,:] = timeArray
#     print(i)
# positionArray = np.array(positionArray)
# timeMat = np.hstack(timeMat)
# Data = np.hstack(positionArray)
# #timeArray = timeArray*10000
# hdf5file2 = h5py.File('predictiveValues.h5', 'a')
# WriteData(hdf5file2, 'data/predictiveValues', Data)
# WriteData(hdf5file2, 'data/time', timeMat)
# fig = MakeFigure(450, 1)
# ax = plt.gca()
# #ax.set_title('Harmonic Predictive Model', fontsize = 12)
# ax.set_xlabel('Time (s)', fontsize = 30)
# ax.set_ylabel('Position (m)', fontsize = 30)
# hist = ax.hist2d(timeMat, Data, normed=True, bins = (400,400), cmax= 0.3, cmap = plt.cm.viridis)
# plt.colorbar(hist[3], ax=ax)
# plt.show()

filename = 'predictiveValues.h5'
hdf5file = h5py.File(filename, 'r')
timeMat = hdf5file['data/time'].value
Data = hdf5file['data/predictiveValues'].value
fig = MakeFigure(450, 1)
ax = plt.gca()
#ax.set_title('Harmonic Predictive Model', fontsize = 30)
ax.set_xlabel('Time (s)', fontsize = 30)
ax.set_ylabel('Position (m)', fontsize = 30)
hist = ax.hist2d(timeMat, Data, normed=True, bins = (1000,1000), cmax= 0.4, cmap = plt.cm.viridis)
#plt.colorbar(hist[3], ax=ax)
plt.show()
