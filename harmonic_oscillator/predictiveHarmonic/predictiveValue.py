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

    xvec = []

        #assert(solver.successful())
    solver.integrate(time[1])

    xvec = solver.y[0]

    return xvec
#-------------------main code------------------
t0 = 0
T = 100
t_interest = 6
time = [t0, t_interest]
trueValue = 0.29834529433859364
spacingMean = 100000
state0 = [1.0, 0.0] # inital state
theta_1 = theta_1[1:]
theta_2 = theta_2[1:]
positionArray = [0]*len(theta_1)
meanValues = [0]*int(len(theta_1)/spacingMean)
iterationLocation = [0]*int(len(theta_1)/spacingMean)
k = 1
integerMultiple = 0
for i in range(0, len(theta_1)):
    #print(positionArray)
    #print(ForwardModel(time, [theta_1[1], theta_2[1]], state0))
    positionArray[i] = ForwardModel(time, [theta_1[i], theta_2[i]], state0)
    k = k + 1
    if (k == spacingMean):
        k = 1
        integerMultiple = integerMultiple + 1
        print(integerMultiple)
        value = np.mean(positionArray[0:(spacingMean*integerMultiple - 1)])
        meanValues[integerMultiple-1] = (value-trueValue)**2
        iterationLocation[integerMultiple-1] = spacingMean*integerMultiple
print(iterationLocation)
print(meanValues)
fig1 = plt.figure(1)
plt.loglog(iterationLocation, meanValues)
plt.title('Central Limit Theorem LogLog Plot')
plt.xlabel('Iterations')
plt.ylabel('E[X(t*)] t* = 6 secs')
plt.draw()
plt.show()
