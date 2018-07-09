# Sample position values of harmonic oscillator at various times

import random
import math
import matplotlib.pyplot as plt

def add_noise(x0, springconst, mass, obs, maxtime):

    # List comprehension to fill a list with 'obs' random numbers. The 
    # list is then sorted. Numbers are rounded to 1 decimal place.
    time = [round(random.uniform(0, maxtime), 1) for _ in range(obs)]
    time.sort()

    # Fill the position vector
    pos = []
    for t in time:
        pos.append(x0*math.cos(math.sqrt(springconst/mass)*t))

    # print(pos)
    # print(time)

    # Plot position wrt time as discrete points
    fig, axs = plt.subplots(nrows = 1, ncols = 1)
    axs.plot(time, pos, 'ro')
    plt.show()

add_noise(1.0, 1.0, 1.0, 10, 20)
