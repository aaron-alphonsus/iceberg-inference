# Sample position values of harmonic oscillator at various times and add 
# gaussian noise

import random
import math
import matplotlib.pyplot as plt

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

    # print(pos)
    # print(time)

    # Plot position (red circles) and noisy position data (blue stars) wrt time
    fig, axs = plt.subplots(nrows = 1, ncols = 1)
    axs.plot(time, pos, 'ro')
    axs.plot(time, noisydata, 'b*')
    plt.show()

add_noise(1.0, 0.5, 10, 1.0, 1.0)
