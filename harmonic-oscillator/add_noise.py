# Sample position values of harmonic oscillator at various times and add 
# gaussian noise. Save data to netCDF file.

import json
import math
import matplotlib.pyplot as plt
import netCDF4
import numpy as np

import random

def add_noise(x0, interval, maxtime, springconst, mass): 
    # Fill time and position vectors beginning at 0.0 and taking observations at
    # each interval
    t = 0.0
    time = []
    pos = []
    noisydata = []
    while t <= maxtime:
        time.append(t)

        p = x0*math.cos(math.sqrt(springconst/mass)*t)
        pos.append(p)
        noisydata.append(p + random.gauss(0, 0.5)) 

        t += interval 

    # Write data to netCDF file format
    timenp = np.asarray(time)
    posnp = np.asarray(pos)
    noisydatanp = np.asarray(noisydata)

    ncfile = netCDF4.Dataset('data.nc', mode='w', format='NETCDF4_CLASSIC')
    # print(ncfile)
    time_dim = ncfile.createDimension('time', None)
    pos_dim = ncfile.createDimension('pos', 21)
    noisydata_dim = ncfile.createDimension('noisydata', 21)
    # for dim in ncfile.dimensions.items():
    #     print(dim)
    ncfile.title = 'Noisy Data'
    
    timecdf = ncfile.createVariable('time', np.float32, ('time',))
    timecdf[:] = timenp
    poscdf = ncfile.createVariable('pos', np.float32, ('pos',))
    poscdf[:] = posnp
    noisydatacdf = ncfile.createVariable('noisydata', np.float32, 
            ('noisydata',))
    noisydatacdf[:] = noisydatanp
    
    # print(timecdf)
    # print(poscdf)
    # print(noisydatacdf)
   
    ncfile.close()


    # print(timenp)
    # print(time)

    # h5f = h5py.File('data.h5', 'w')
    # h5f.create_dataset('dataset_1', data=pos)
    # h5f.close()

    # # Write data to file using json
    # data = {'time'      : time,
    #         'pos'       : pos,
    #         'noisydata' : noisydata}
    # with open('data.txt', 'w') as f:
    #     json.dump(data, f, ensure_ascii=False)

    # print(time)
    # print(pos)
    # print(noisydata)

    # # Plot position (red circles) and noisy position data (blue stars) wrt time
    # fig, axs = plt.subplots(nrows = 1, ncols = 1)
    # axs.plot(time, pos, 'ro')
    # axs.plot(time, noisydata, 'b*')
    # plt.show()

add_noise(1.0, 0.5, 10, 1.0, 1.0)
