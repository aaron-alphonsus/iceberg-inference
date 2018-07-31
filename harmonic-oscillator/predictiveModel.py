#---------------modules--------------------
import numpy as np
import math as math
from scipy import stats
import h5py
import matplotlib.pyplot as plt
from matplotlib import rcParams
from math import *
from MakeFigure import *
#from ForwardModel import *
from scipy.integrate import ode
#----------read data--------------------------------
filename = 'data_mcmc.h5'
hdf5file = h5py.File(filename, 'r')
theta_1 = hdf5file['data_mcmc/theta1'].value
theta_2 = hdf5file['data_mcmc/theta2'].value
#-----------forward model code------------------------
def rhs(t, state, theta):
    x = state[0]
    u = state[1]

    omega = theta[0]
    gamma = theta[1]

    return [u, -omega*omega*x-gamma*u]

def jacobian(t, state, theta):
    omega = theta[0]
    gamma = theta[1]

    return [[0.0, 1.0],[-omega*omega, -2.0*gamma*omega]]

# time - times to observe numerical solution of the damped harmonic oscilator
# theta - parameters: [\omega, \gamma]
def ForwardModel(time, theta, state0):
    # create a solver
    solver = ode(rhs, jacobian)

    # set the numerical options (e.g., method and tolerances)
    solver.set_integrator('vode', method='bdf', with_jacobian=True) # play with tolerances?

    solver.set_initial_value(state0, time[0])
    solver.set_f_params(theta)
    solver.set_jac_params(theta)

    xvec = [state0[0]]

    for t in time[1:]:
        #assert(solver.successful())
        solver.integrate(t)

        xvec = xvec+[solver.y[0]] # only need the real part

    return xvec
#-------------------main code------------------
T = 10 # observation points for position
t0 = 0.0
tf = 10.0
state0 = [1.0, 0.0] # inital state
positionArray = [[0]*T]*len(theta_1)
timeArray = [[0]*T]*len(theta_1)
for i in range(0, len(theta_1)):
    timeArray[i] = sorted(np.random.uniform(t0, tf, size=T))
    positionArray[i] = ForwardModel(timeArray[i], [theta_1[i], theta_2[i]], state0)
timePlot = np.hstack(timeArray)
positionPlot = np.hstack(positionArray)
# plt.hist(positionPlot, bins=100)
# plt.show()
#print(positionPlot)
fig = MakeFigure(450, 1)
ax = plt.gca()
ax.set_title('2D Histogram', fontsize = 12)
ax.set_xlabel('Time (s)', fontsize = 10)
ax.set_ylabel('Position (m)', fontsize = 10)
hist = ax.hist2d(timePlot, positionPlot, normed=True, bins = (25,25), cmap = plt.cm.viridis)
plt.colorbar(hist[3], ax=ax)
plt.show()
