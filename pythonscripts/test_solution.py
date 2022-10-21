import matplotlib.pyplot as plt
import numpy as np
from utilities_monolithic import *
from utilities_staggered import *


## Input parameters
#
m  = 1.0;                        # total mass, m=ms+mf
k  = 4.0*np.pi**2;               # stiffness
c  = 0.0;                        # damping coefficient

w  = np.sqrt(k/m);               # (circular) natural frequency
xi = c/(2.0*np.sqrt(k*m));       # damping ratio
wd = w*np.sqrt(1.0-xi*xi);       # damped natural frequency
T  = 2.0*np.pi/w;                # time period

dt = T/100.0;

timesteparray = np.arange(0.0, 5*T+dt, dt)

mr = 1.0;                       # mass ratio


alpha = mr/(1.0+mr);
beta  = alpha;
#beta  = 1.0;


# solution with the monolithic scheme
dispMono, veloMono = solution_monolithic_BE(m, c, k, mr, timesteparray)

# solution with staggered scheme
#dispNumfP1, veloNumfP1 = solution_staggered_forcerelaxation_BE(m, c, k, mr, predType, betafactor, timesteparray)

dispNumfP1, veloNumfP1 = solution_staggered_forcerelaxation_BE(m, c, k, mr, 2, beta, timesteparray)
#dispNumfP1, veloNumfP1 = solution_staggered_forcerelaxation_BE(m, c, k, mr, 2, 1.3, timesteparray)

beta  = alpha/(2-3*alpha)/5;
dispNumfP2, veloNumfP2 = solution_staggered_displacementrelaxation_BE(m, c, k, mr, 0, beta, timesteparray)
#dispNumfP3, veloNumfP3 = solution_staggered_displacementrelaxation_BE(m, c, k, mr, 1, beta, timesteparray)
#dispNumfP4, veloNumfP4 = solution_staggered_displacementrelaxation_BE(m, c, k, mr, 2, beta, timesteparray)


plt.plot(timesteparray, dispMono,'g', linewidth=4)
plt.plot(timesteparray, dispNumfP1,'k', linewidth=2)
plt.plot(timesteparray, dispNumfP2,'b--')
#plt.plot(timesteparray, dispNumfP3,'r--')
#plt.plot(timesteparray, dispNumfP4,'m--')
plt.xlabel("Time", fontsize=14)
plt.ylabel("Displacement", fontsize=14)
plt.xlim([min(timesteparray), max(timesteparray)])
plt.ylim([-1.5, 1.5])
plt.grid()
plt.show()