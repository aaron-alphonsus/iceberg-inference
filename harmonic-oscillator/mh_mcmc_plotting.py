import h5py

from MakeFigure import *

# Read data from file
filename = 'data_mcmc.h5'
hdf5file = h5py.File(filename, 'r')
t1 = hdf5file['data_mcmc/theta1'].value
t2 = hdf5file['data_mcmc/theta2'].value
acc_rate = hdf5file['data_mcmc/acc_rate'].value

print("Acceptance rate = ", acc_rate)

# Mixing graphs for theta1 and theta2
fig = MakeFigure(450, 1)
ax = plt.gca()
ax.set_title('Spring Coefficient Mixing', fontsize = 12)
ax.set_xlabel('Iterations', fontsize = 10)
ax.set_ylabel('Spring Coefficient ($k/m$)', fontsize = 10)
ax.plot(t1)

fig = MakeFigure(450, 1)
ax = plt.gca() 
ax.set_title('Damping Coefficient Mixing', fontsize = 12)
ax.set_xlabel('Iterations', fontsize = 10)
ax.set_ylabel('Damping Coefficient ($c/m$)', fontsize = 10)
ax.plot(t2)

# Scatter Plot
fig = MakeFigure(450, 1)
ax = plt.gca() 
ax.set_title('Scatter Plot', fontsize = 12)
ax.set_xlabel('Spring Coefficient ($k/m$)', fontsize = 10)
ax.set_ylabel('Damping Coefficient ($c/m$)', fontsize = 10)
ax.scatter(t1, t2)

# 2D Histogram
fig = MakeFigure(450, 1)
ax = plt.gca()
ax.set_title('2D Histogram', fontsize = 12)
ax.set_xlabel('Spring Coefficient ($k/m$)', fontsize = 10)
ax.set_ylabel('Damping Coefficient ($c/m$)', fontsize = 10)
hist = ax.hist2d(t1, t2, bins = (100,100), cmin = 1, cmap = plt.cm.inferno)
plt.colorbar(hist[3], ax=ax)
plt.show()
