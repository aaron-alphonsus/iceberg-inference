# Sample position values of harmonic oscillator at various times and add 
# gaussian noise. Save data to netCDF file.

# import json
import math
import netCDF4 as nc
import numpy as np
import random

def add_noise(x0, interval, maxtime, springconst, mass): 
    # Fill time, position, and 'noisydata' vectors beginning at 0.0 and taking 
    # observations at each interval. 
    t = 0.0
    time = []
    pos = []
    noisydata = []
    while t <= maxtime:
        time.append(t)

        p = x0*math.cos(math.sqrt(springconst/mass)*t)
        pos.append(p)
        # The noisydata vector contains the position values along with additive 
        # noise from a gaussian distribution
        noisydata.append(p + random.gauss(0, 0.5)) 

        t += interval 

    # print(time, pos, noisydata, sep = '\n\n')

    # Convert arrays to numpy arrays
    timenp = np.asarray(time)
    posnp = np.asarray(pos)
    noisydatanp = np.asarray(noisydata)

    # Open a new netCDF file in writing mode
    ncfile = nc.Dataset('data.nc', mode='w', format='NETCDF4_CLASSIC')
    
    # Define dimensions of the data
    time_dim = ncfile.createDimension('time', None)
    pos_dim = ncfile.createDimension('pos', 21)
    noisydata_dim = ncfile.createDimension('noisydata', 21)
    
    ncfile.title = 'Noisy Data'
   
    # Create variables and fill them with the numpy arrays
    timecdf = ncfile.createVariable('time', np.float32, ('time',))
    timecdf[:] = timenp
    poscdf = ncfile.createVariable('pos', np.float32, ('pos',))
    poscdf[:] = posnp
    noisydatacdf = ncfile.createVariable('noisydata', np.float32, 
            ('noisydata',))
    noisydatacdf[:] = noisydatanp
   
    # Closing the file writes the variables to the file
    ncfile.close()
    
    # # Write data to file using json
    # data = {'time'      : time,
    #         'pos'       : pos,
    #         'noisydata' : noisydata}
    # with open('data.txt', 'w') as f:
    #     json.dump(data, f, ensure_ascii=False)
 
add_noise(1.0, 0.5, 10, 1.0, 1.0)
