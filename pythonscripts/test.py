#from cProfile import label
import matplotlib.pyplot as plt
import numpy as np
from utilities_monolithic import *
from utilities_staggered import *


## Input parameters
#
m  = 1.0;                        # total mass, m=ms+mf
k  = 4.0*np.pi**2;               # stiffness
c  = 0.0;                        # damping coefficient


mr = 5;                       # mass ratio

alpha = mr/(1.0+mr);
beta  = alpha;
#beta  = alpha/2.0;
beta  = 1.0;


logtimestep = np.linspace(-4, 3, 101)


specRadMono = spectralRadius_monolithic_BE(m, c, k, mr, logtimestep)
#predictor_type = 2
#specRad1 = spectralRadius_staggered_forcerelaxation_BE(m, c, k, mr, predictor_type, beta, logtimestep)

predictor_type = 2
beta  = alpha/(2-3*alpha)/2;
specRad2 = spectralRadius_staggered_displcementrelaxation_BE(m, c, k, mr, 0, beta, logtimestep)
beta  = alpha/(3-3*alpha)/200;
specRad3 = spectralRadius_staggered_displcementrelaxation_BE(m, c, k, mr, 1, beta, logtimestep)
#beta  = alpha;
specRad4 = spectralRadius_staggered_displcementrelaxation_BE(m, c, k, mr, 2, beta, logtimestep)


plt.plot(logtimestep, specRadMono, 'g', linewidth=4, label="Monolithic")
plt.plot(logtimestep, specRad2, 'k', linewidth=2, label=r"$d^{P0}$")
plt.plot(logtimestep, specRad3, 'r', linewidth=2, label=r"$d^{P1}$")
plt.plot(logtimestep, specRad4, 'b', linewidth=2, label=r"$d^{P2}$")
plt.xlabel(r"$\Delta t/T$", fontsize=14)
plt.ylabel(r"Spectral Radius ($\rho$)", fontsize=14)
plt.xlim([min(logtimestep), max(logtimestep)])
plt.ylim([0.0, 1.2])
plt.legend()
plt.grid()
plt.show()