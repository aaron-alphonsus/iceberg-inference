#---------------modules--------------------
import numpy as np
import math as math
from scipy import stats
import h5py
import matplotlib.pyplot as plt
from matplotlib import rcParams
from math import *
from MakeFigure import *
from IcebergForward import *
from scipy.integrate import ode
#----------read data--------------------------------
filename = 'AaronData.h5'
hdf5file = h5py.File(filename, 'r')
theta_1 = hdf5file['data_mcmc/theta1'].value
theta_2 = hdf5file['data_mcmc/theta2'].value
theta_1 = theta_1[89999:99999]
theta_2 = theta_2[89999:99999]
#-------------------main code------------------
def WriteData(hdf5file, name, data):
    if name in hdf5file:
        del hdf5file[name]
    hdf5file.create_dataset(name, data=data)
state0 = [0, -1, 0, 0] # inital state
timeArray = list(range(100))
T = len(timeArray)
positionArray = [[0, 0]*T]*len(theta_1)
for i in range(0, len(theta_1)):
    positionArray[i] = ForwardModel(timeArray, [theta_1[i], theta_2[i]], state0)
    print(i)
positionArray = np.array(positionArray)
xData = np.hstack(positionArray[:,:,0])
yData = np.hstack(positionArray[:,:,1])
hdf5file2 = h5py.File('predictiveValues.h5', 'a')
WriteData(hdf5file2, 'data/predictiveX', xData)
WriteData(hdf5file2, 'data/predictiveY', yData)
fig = MakeFigure(450, 1)
ax = plt.gca()
ax.set_title('Iceberg Predictive Model', fontsize = 12)
ax.set_xlabel('Latitude (deg)', fontsize = 10)
ax.set_ylabel('Longitude (deg)', fontsize = 10)
hist = ax.hist2d(xData, yData, normed=True, bins = (400,400), cmax= 0.2, cmap = plt.cm.viridis)
plt.colorbar(hist[3], ax=ax)
plt.show()
