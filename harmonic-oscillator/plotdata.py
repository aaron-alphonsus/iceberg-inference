# Read harmonic oscillator data from netCDF file and plot it

import netCDF4 as nc
import matplotlib.pyplot as plt

# Open datafile with the time, position, and 'noisydata'
data = nc.Dataset('data.nc')

# Plot data using matplotlib
fig, axs = plt.subplots(nrows = 1, ncols = 1)
axs.plot(data.variables['time'][:], data.variables['pos'][:], 'ro')
axs.plot(data.variables['time'][:], data.variables['noisydata'][:], 'b*')
plt.show()
