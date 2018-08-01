# Author: Andrew Davis
# Edited by: Aaron Alphonsus

from scipy.integrate import ode

# Implement the RHS (and Jacobian) of a damped, forced harmonic oscillator
#
# dx/dt = u
# du/dt = F(u,x,t)/m - \gamma u - \omega^2 x
#
# spring constant: k, damping coefficent: c, mass: m
# \omega = \sqrt(k/m)
# \gamma = c/m

# t - time
# x - unknown; solve for this!
# theta - [\omega, \gamma] 
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
    solver.set_integrator('zvode', method='bdf', with_jacobian=True, 
            atol=1.0e-8, rtol=1.0e-8) # play with tolerances?

    solver.set_initial_value(state0, time[0])
    solver.set_f_params(theta)
    solver.set_jac_params(theta)
    
    xvec = [state0[0]]

    for t in time[1:]:
        assert(solver.successful())
        solver.integrate(t)
        
        xvec = xvec+[solver.y[0].real] # only need the real part

    return xvec
