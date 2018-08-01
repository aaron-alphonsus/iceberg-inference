# Javier Salazar & Aaron Alphonsus
# Skeleton Code: James Ronan
# Iceberg Forward Model Rev 2.0
#------------libraries-------------------------------
from scipy.integrate import ode # ode system
import numpy as np # array manipulation
import matplotlib.pyplot as plt #plotting
import math # calculations
from scipy.interpolate import interpn # for latitude/longitude grid
import h5py # read files
#import matplotlib.cm as cm
#-------------parameters---------------------------------------57 43.8 0.01
x0,y0,u0,v0 = 304, 61, 0.0, 0.0 # postion and intial velocity for iceberg
inferValues = [1.0, 0.1] # c_water and c_air
plotOption = 1 # 0 -> no image overlay. 1 -> image background
resolutionData = 88
#----------calculations-----------------------------------
xvec = [[x0,y0]] # first value
def plotData(pointPairs, plotOption, errorCode):
    xData=[x[0] for x in pointPairs]
    yData=[x[1] for x in pointPairs]
    if (plotOption == 0):
        plt.plot(xData,yData)
    if (plotOption == 1):
        img = plt.imread("land_mass.png") # image for scatter background
        plt.imshow(img)
        xData2 = np.interp(xData, [lon_current[0], lon_current[-1]], [0, len(lon_current)]) # linearly map lat/lon values to image pixel locations
        yData2 = np.interp(yData, [lat_current[0], lat_current[-1]], [0, len(lat_current)])
        yData2[:] = [len(lat_current)-x for x in yData2]
        yticks= np.array(np.linspace(lat_current[-1],lat_current[0],10))
        ylocs = np.array(np.linspace(0, len(lat_current), 10))
        yticks = np.around(yticks, decimals=1)
        xticks= np.array(np.linspace(lon_current[0],lon_current[-1],10))
        xlocs = np.array(np.linspace(0, len(lon_current), 10))
        xticks = np.around(xticks, decimals=1)
        plt.plot(xData2,yData2)
        plt.xticks(xlocs, xticks)
        plt.yticks(ylocs, yticks)
    plt.title('Iceberg Predicted Path')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    if (errorCode == 1):
        print("Program finished early. Out of bounds of data region. Interpolation error.")
    if (errorCode == 3):
        print("Program finished early. Object touched land. Nan interpolation error.")
    plt.show()
def checkValidPoints(x,y):
    global u_matrix_ocean
    global lon_current
    global lat_current
    if (x < rangeLon[0] or x > rangeLon[1] or y < rangeLat[0] or y > rangeLat[1]):
        global xvec
        plotData(xvec, plotOption, 1)
        quit()
    lon_current = np.asarray(lon_current)
    lat_current = np.asarray(lat_current)
    idx = (np.abs(lon_current - x)).argmin()
    idy = (np.abs(lat_current - y)).argmin()
    if (math.isnan(u_matrix_ocean[0,idy,idx]) == True):
        plotData(xvec, plotOption, 3)
        quit()
#------------input stored data and process-----------------------
def fixGiantMess(filename):
    hdf5file = h5py.File(filename, 'r')
    lat_current= np.array(hdf5file['current_variables/lat'].value) #collect information
    lon_current= np.array(hdf5file['current_variables/lon'].value)
    lat_wind= np.array(hdf5file['wind_variables/lat'].value)
    lon_wind= np.array(hdf5file['wind_variables/lon'].value)
    time_wind= np.array(hdf5file['wind_variables/time'].value)
    time_current= np.array(hdf5file['current_variables/time'].value)
    time_combined = time_wind[:,1] # sync different times
    time_combined = time_combined[42:86]
    speed_wind_u= np.array(hdf5file['wind_variables/u_wind1'].value)
    speed_current_u= np.array(hdf5file['current_variables/u_current1'].value)
    speed_wind_v= np.array(hdf5file['wind_variables/v_wind1'].value)
    speed_current_v= np.array(hdf5file['current_variables/v_current1'].value)
    speed_wind_u = speed_wind_u[42:86,:,:] # sync speed info for combined time
    speed_wind_v = speed_wind_v[42:86,:,:]
    speed_current_u = speed_current_u[0:44,:,:]
    speed_current_v = speed_current_v[0:44,:,:]
    return[time_combined, speed_current_u, speed_current_v, speed_wind_u, speed_wind_v, lat_current, lon_current, lat_wind, lon_wind]
#-----------convert meters/sec to deg/sec--------------------------------------------------
def convertUnits(u, v, x, y): # not currently used in code
    length_v = 111320.0 # for latitude the length is fixed per degree
    v_new = 3.6*v/length_v # veolicty is deg/sec
    # meters per degree is a function for longitude that depends on latitude
    length_u = float(40075000.0*math.degrees(math.cos(math.radians(y)))/360.0)
    u_new = 3.6*u/length_u # constant is related to length at equator
    #u_new = u
    #v_new = v
    return [u_new, v_new]
#--------------coriolis force function------------------------------
def Fcor(state, t):
    y = state[1] # y position degree latitude
    [u_wind, v_wind] = LookUpAir(state[0], state[1], t) # wind speed
    latitude_rad = (math.pi/180)*y # convert to radian
    coriolis_freq = 2*(7.2921*(10**-5))*math.sin(latitude_rad) # coriolis freq/paremeter/coefficient
    coriolisforce_permass_x = -coriolis_freq*u_wind # per equation given velocity as meters/s
    coriolisforce_permass_y = -coriolis_freq*v_wind
    return [coriolisforce_permass_x, coriolisforce_permass_y] # Force meter/s^2
#------------current speeds using location
def LookUpWater(state, t):
    x = state[0] # latitude/longitude info
    y = state[1]
    checkValidPoints(x,y)
    if(t > timeRange[-1]): # ode goes over range of time by 0.01 for example so time is capped at final time
         t = timeRange[-1]
    u_point = interpn((timeData, lat_current , lon_current), u_matrix_ocean, np.array([t, y, x]).T) # interpolate from grid info
    v_point = interpn((timeData, lat_current , lon_current), v_matrix_ocean, np.array([t, y, x]).T)
    [u_convert, v_convert] = convertUnits(u_point[0], v_point[0], x, y) # convert units to deg info
    return [u_convert,v_convert]
#--------------air interpolation-------------------------
def LookUpAir(x,y, t):
    if(t > timeRange[-1]): # same as look up water
         t = timeRange[-1]
    u_point2 = interpn((timeData, lat_wind , lon_wind), u_matrix_wind, np.array([t, y, x]).T) # interpolate using linear
    v_point2 = interpn((timeData, lat_wind , lon_wind), v_matrix_wind, np.array([t, y, x]).T)
    [u_convert, v_convert] = convertUnits(u_point2[0], v_point2[0], x, y) # convert units
    return [u_convert,v_convert]
#----------water force computation--------------------
def Fwater(state,c_w, t): # water force
    u_ocean=np.array(LookUpWater(state, t)) # get from ocean currents
    u_berg=np.array([state[2],state[3]])
    u_relw=np.subtract(u_ocean,u_berg) # subtract iceberg velocity to get relative
    nuw=np.linalg.norm(u_relw) # l2 norm
    return c_w*nuw*u_relw
#------------air force computation-------------------------
def F_air(state,c_a, t):
    u_air=np.array(LookUpAir(state[0],state[1], t)) # same as above
    u_berg=np.array([state[2],state[3]])
    u_rela=np.subtract(u_air,u_berg)
    nua=np.linalg.norm(u_rela)
    return c_a*nua*u_rela
#-----------rhs ode model----------------------------
def rhs(t, state, theta):
    x = state[0] # x position
    y = state[1] # y position
    u = state[2] # x velocity
    v = state[3] # y velocity
    c_water = theta[0] # set inference values
    c_air = theta[1]
    Water_F=Fwater(state,theta[0], t) # get forces
    Air_F=F_air(state,theta[1], t)
    Cor_F=Fcor(state, t)
    return [u,v, Water_F[0]+Air_F[0]+Cor_F[0],Water_F[1]+Air_F[1]+Cor_F[1]]
#-------------------forward model given paramters of inference-------------
def ForwardModel(time, theta, state0):
    global xvec
    solver = ode(rhs)
    solver.set_integrator('vode', method='bdf', with_jacobian=False) #set steps, ode method, etc.. Nonstiff only
    solver.set_initial_value(state0, time[0])
    solver.set_f_params(theta)
    for t in time[1:]:
        #assert(solver.successful())
        solver.integrate(t)
        x = solver.y[0]
        y = solver.y[1]
        checkValidPoints(x, y)
        xvec = xvec+[[x,y]]
    return xvec
#------------------global calculations used in functions-----------------------
intialState = [x0, y0, u0, v0]
[timeArray, u_matrix_ocean, v_matrix_ocean, u_matrix_wind, v_matrix_wind, lat_current, lon_current, lat_wind, lon_wind] = fixGiantMess('IcebergData.h5')
timeRange = np.linspace(0,len(timeArray)-1,resolutionData)
timeData = np.array(range(len(timeArray)))
# print(timeRange)
# print(timeArray)
# print(timeData)
#timeRange = np.array(range(0,len(timeArray)))
rangeLat = [max(lat_wind[0], lat_current[0]), min(lat_wind[-1], lat_current[-1])]
rangeLon = [max(lon_wind[0], lon_current[0]), min(lon_wind[-1], lon_current[-1])]
#-------------main function---------------------------------
ObsData=ForwardModel(timeRange,inferValues,intialState)
#print(ObsData)
# matrix = u_matrix_ocean[0,:,:]
# matrixLand = np.isnan(matrix)
# plt.imsave('land_mass2.png', np.array(~matrixLand), cmap=cm.gray)
plotData(ObsData, plotOption, 0)
