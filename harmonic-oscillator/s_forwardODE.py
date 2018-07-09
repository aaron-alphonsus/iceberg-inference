# Description: Damped harmonic oscillator using scipy's ODE solver

# Author: Andrew Davis
# Edited by: Aaron Alphonsus

from scipy.integrate import ode
import matplotlib.pyplot as plt
import math

# t - time (not used)
# x - unknown; solve for this!
# kom - k over m; k/m (spring constant: k, mass: m)
# com - c over m; c/m (damping coefficient: c)
# def rhs(t, state, kom):
def rhs(t, state, kom, com):
    x = state[0]
    u = state[1]

    # return [u, -kom*x]
    return [u, -kom*x-com*u]

# def jacobian(t, state, kom):
def jacobian(t, state, kom, com):
    # x = state[0]
    # u = state[1]


    # return [[0.0, 1.0],[-kom, 0]]
    return [[0.0, 1.0],[-kom, -com]]
    
################################ MAIN PROGRAM ##################################

# initial conditions
x0     = 1.0
u0     = 0.0
state0 = [x0, u0] # initial state
t0     = 0.0      # initial time

# constants
k = 1.0
m = 1.0
c = 0.225

# create a solver
solver = ode(rhs, jacobian)
# l = dir(solver)
# print(l)

# set the numerical options (e.g. method and tolerances)
solver.set_integrator('zvode', method='bdf', with_jacobian=True, atol=1.0e-2, 
        rtol=1.0e-2) # play with tolerances?

solver.set_initial_value(state0, t0)
solver.set_f_params(k/m, c/m)
solver.set_jac_params(k/m, c/m)

# print(solver.f_params)
# print(solver.jac_params)

tf = 20.0 # final time
dt = 0.5

time  = [0.0]
xvec  = [state0[0]]
xtrue = [x0]
uvec  = [state0[1]]
utrue = [0.0]

# print("%g %g %g" % (solver.t, solver.y[0], solver.y[1]))
while solver.successful() and solver.t < tf:
    solver.integrate(solver.t + dt)
    
    time  = time + [solver.t]
    xvec  = xvec + [float(solver.y[0])]
    xtrue = xtrue+[x0*math.cos(math.sqrt(k/m)*solver.t)]
    uvec  = uvec+[float(solver.y[1])]
    utrue = utrue+[-x0*math.sqrt(k/m)*math.sin(math.sqrt(k/m)*solver.t)]

# print(time, xvec, xtrue, uvec, sep='\n\n')
# print(len(time), len(xvec), len(xtrue), len(uvec))

fig, axs = plt.subplots(nrows=2,ncols=1)
axs[0].plot(time, xvec)
# axs[0].plot(time, xtrue)
axs[1].plot(time, uvec)
# axs[1].plot(time, utrue)

plt.show()
