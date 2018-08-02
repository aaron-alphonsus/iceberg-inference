# Author: James Ronan

from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt

# Implement the simplified model of iceberg movement,
# DE from F=ma with the forces we are looking at being
# the coriolis effect, water drag, and wind drag


# theta - [c_water, c_air]
Ccor=0
def Fcor(state):
    #This is the Coriolis force or force/mass
    #right now it is a constant function, but could be a y dependent function.
    # takes argument state, which is [x,y,u,v]
    # Does the coriolis effect affect both x and y?
    return np.array([-1,Ccor])

def LookUpWater(x,y):
    # This function should look up the
    # ocean current information at this position
    # Right now I don't have it
    return [1,1]

def MUScurrent(x,y):
    # Made up swirly current
    R=x*x+y*y+5
    return [10*y/R,-10*x/R]


def LookUpAir(x,y):
    #This function should look up the
    #wind information
    #right now nothing is here.
    return [.2,.2]

def Fwater(state,c_w):
    #This function will first look up the ocean current at the
    # point in water where we are, and then
    # use the relative velocity of the water
    # to return c_w*|u_rel|*u_rel
    u_ocean=np.array(MUScurrent(state[0],state[1]))
    u_berg=np.array([state[2],state[3]])
    u_relw=np.subtract(u_ocean,u_berg)
    nuw=np.linalg.norm(u_relw)
    return c_w*nuw*u_relw

def F_air(state,c_a):
    #look up the wind, find relative vels, then determine the force vector
    u_air=np.array(LookUpAir(state[0],state[1]))
    u_berg=np.array([state[2],state[3]])
    u_rela=np.subtract(u_air,u_berg)
    nua=np.linalg.norm(u_rela)
    return c_a*nua*u_rela

def rhs(t, state, theta):
    #Right now, we have constant wind and water currents,
    # so we are forming an autonomous system, however if we incorporate
    #changing currents the t will be necessary
    x = state[0] # x position
    y = state[1] # y position
    u = state[2] # x velocity
    v = state[3] # y velocity
    c_water = theta[0]
    c_air = theta[1]
    Water_F=Fwater(state,theta[0])
    Air_F=F_air(state,theta[1])
    Cor_F=Fcor(state)
    return [u,v, Water_F[0]+Air_F[0]+Cor_F[0],Water_F[1]+Air_F[1]+Cor_F[1]]

# time - times to observe the numerical solution
# theta - parameters: [c_water, c_air]
def ForwardModel(time, theta, state0):
    #time should be a list of increasing times, this is where the
    # model will output points
    # Theta should be [c_w,c_a]
    # State0 is initial conditions, [x,y,u,v]
    # create a solver
    solver = ode(rhs)
    # set the numerical options (e.g., method and tolerances)
    solver.set_integrator('vode', method='bdf', with_jacobian=False, atol=1.0e-2, rtol=1.0e-4)

    solver.set_initial_value(state0, time[0])
    solver.set_f_params(theta)
    xvec = [[state0[0],state0[1]]]

    for t in time[1:]:
        assert(solver.successful())
        solver.integrate(t)

        xvec = xvec+[[solver.y[0],solver.y[1]]]

    return xvec

if __name__ == '__main__':
    ObsTime=list(range(100))
    ObsData=ForwardModel(ObsTime, [1.5, 1.5], [0, -1, 0, 0])
    Xdata = [x[0] for x in ObsData]
    Ydata = [x[1] for x in ObsData]
    nx, ny = 64,64
    xgrid = np.linspace(-16, 16, nx)
    ygrid = np.linspace(-16, 16, ny)
    X, Y = np.meshgrid(xgrid, ygrid)
    U, V = MUScurrent(X, Y)
    plt.quiver(X, Y, U, V)
    plt.plot(Xdata, Ydata)
    plt.show()
