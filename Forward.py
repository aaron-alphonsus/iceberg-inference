# Javier Salazar & Aaron Alphonsus
# Skeleton Code: James Ronan
# Iceberg Forward Model Rev 1.0
#------------libraries-------------------------------
from scipy.integrate import ode
from scipy.ndimage import map_coordinates
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import math
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.interpolate import interpn
#-------------parameters---------------------------------------
upperLatitudeBound = 150
lowerLatitudeBound = 0
upperLongitudeBound = 1000
lowerLongitudeBound = 800
x0,y0,u0,v0 = 320.0, 46.666666, -0.371430489,0.123941015917
inferValues = [1, 1]
ti = 0
tf = 37
#------------------global calculations used in functions-----------------------
timeRange = np.array(range(ti,tf))
intialState = [x0, y0, u0, v0]
data = nc.Dataset("oscar_vel2018.nc")
u_matrix = data.variables['u'][ti:(tf),0,lowerLatitudeBound:upperLatitudeBound,lowerLongitudeBound:upperLongitudeBound] # all time not just zero implement later
v_matrix = data.variables['v'][ti:(tf),0,lowerLatitudeBound:upperLatitudeBound,lowerLongitudeBound:upperLongitudeBound]
latitude = data.variables['latitude'][lowerLatitudeBound:upperLatitudeBound]
longitude = data.variables['longitude'][lowerLongitudeBound:upperLongitudeBound]
latitude2 = np.flipud(latitude)
u_matrix2 = u_matrix
v_matrix2 = v_matrix
for i in timeRange:
    u_matrix2[i,:,:] = np.flipud(u_matrix2[i,:,:])
for i in timeRange:
    v_matrix2[i,:,:] = np.flipud(v_matrix2[i,:,:])
#--------------coriolis force function------------------------------
def Fcor(state):
    y = state[1] # y position degree latitude
    [u_wind, v_wind] = LookUpAir(state[0], state[1]) # wind speed
    latitude_rad = (math.pi/180)*y # convert to radian
    coriolis_freq = 2*(7.2921*(10**-5))*math.sin(latitude_rad) # coriolis freq/paremeter/coefficient
    coriolisforce_permass_x = -coriolis_freq*u_wind # per equation given velocity as meters/s
    coriolisforce_permass_y = -coriolis_freq*v_wind
    return [coriolisforce_permass_x, coriolisforce_permass_y] # Force meter/s^2
#------------current speeds using location
def LookUpWater(state, t):
    x = state[0]
    y = state[1]
    if(t > tf-1):
         t = tf-1
    u_point = interpn((timeRange, latitude2 , longitude), u_matrix2, np.array([t, y, x]).T)
    v_point = interpn((timeRange, latitude2 , longitude), v_matrix2, np.array([t, y, x]).T)
    return [u_point[0],v_point[0]]
#--------------air interpolation-------------------------
def LookUpAir(x,y):
    return [.2,.6]
#----------water force computation--------------------
def Fwater(state,c_w, t):
    u_ocean=np.array(LookUpWater(state, t))
    u_berg=np.array([state[2],state[3]])
    u_relw=np.subtract(u_ocean,u_berg)
    nuw=np.linalg.norm(u_relw)
    return c_w*nuw*u_relw
#------------air force computation-------------------------
def F_air(state,c_a):
    u_air=np.array(LookUpAir(state[0],state[1]))
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
    c_water = theta[0]
    c_air = theta[1]
    Water_F=Fwater(state,theta[0], t)
    Air_F=F_air(state,theta[1])
    Cor_F=Fcor(state)
    return [u,v, Water_F[0]+Air_F[0]+Cor_F[0],Water_F[1]+Air_F[1]+Cor_F[1]]
#-------------------forward model given paramters of inference-------------
def ForwardModel(time, theta, state0):
    solver = ode(rhs)
    solver.set_integrator('vode', method='bdf', with_jacobian=False)
    solver.set_initial_value(state0, time[0])
    solver.set_f_params(theta)
    xvec = [[state0[0],state0[1]]]
    for t in time[1:]:
        assert(solver.successful())
        solver.integrate(t)
        xvec = xvec+[[solver.y[0],solver.y[1]]]
    return xvec
#-------------main function---------------------------------
ObsData=ForwardModel(timeRange,inferValues,intialState)
Xdata=[x[0] for x in ObsData]
Ydata=[x[1] for x in ObsData]
plt.scatter(Xdata,Ydata)
plt.title('Iceberg Predicted Path')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
