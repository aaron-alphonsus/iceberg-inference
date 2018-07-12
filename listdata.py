# Opens oscar_vel2018.nc, reads data in and plots it

# Note: Download oscar_vel2018.nc.gz from 
# ftp://podaac-ftp.jpl.nasa.gov/allData/oscar/preview/L4/oscar_third_deg and 
# unzip the file

# Author: Matthew Parno
# Edited by: Aaron Alphonsus

import netCDF4 as nc
import numpy as np

import matplotlib.pyplot as plt

fn = "oscar_vel2018.nc"
data = nc.Dataset(fn)
print(data)
print(data.variables['latitude'])
print(data.variables['u'])
print(data.variables['v'])
print(data.variables['um'])
print(data.variables['vm'])

fig, axs = plt.subplots(nrows=2, ncols=1)
axs[0].imshow(data.variables['u'][0,0,:150,800:950])
axs[1].imshow(data.variables['u'][0,0,:,:])

plt.show()
