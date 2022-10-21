
###################################################
#
# staggered scheme with backward-Euler scheme
# solid solver first
#
###################################################


import matplotlib.pyplot as plt
import numpy as np
import sys
from pylab import *
from utilities_monolithic import *
from utilities_staggered import *


## Input parameters
#
m  = 1.0;                        # total mass, m=ms+mf
k  = 4.0*np.pi**2;               # stiffness
k  = 1.0;
c  = 0.0;                        # damping coefficient

mr = 3.0;                       # mass ratio
predictor_type = 1;

d0 = 1.0;                        # initial displacement
v0 = 0.0;                        # initial velocity
a0 = (-c*v0 - k*d0)/m;           # initial acceleration

## Intermediate (auxiliary) parameters
#
w  = np.sqrt(k/m);               # (circular) natural frequency
xi = c/(2.0*np.sqrt(k*m));       # damping ratio
wd = w*np.sqrt(1.0-xi*xi);       # damped natural frequency
T  = 2.0*np.pi/w;                # time period


dt = T/200.0;

timesteparray = np.arange(0.0, 5*T+dt, dt)

Nt = np.size(timesteparray);

## parameters 
alpha = mr/(1.0+mr);              # auxiliary parameter
beta  = 1.0;

##
# solution phase starts from here

dispNum  = np.zeros((Nt,1), dtype=float)
veloNum  = np.zeros((Nt,1), dtype=float)
veloFluidNum  = np.zeros((Nt,1), dtype=float)

Klocal = alpha/dt/dt + w*w;

# initialise variables used in the solution
dispPrev = d0;
veloPrev = v0;
accePrev = a0;
veloPrev2 = 0.0;

veloFluidPrev = 0.0;
acceFluidPrev = 0.0;

dispPredPrev  = d0;

disp = 0.0;
velo = 0.0;
acce = 0.0;

# store the solutions at t=0
dispNum[0] = d0;
veloNum[0] = v0;

for ii in range(1,Nt):

    # fluid problem

    if (predictor_type == 0):
        dispPred = dispPrev;
    elif (predictor_type == 1):
        dispPred = dispPrev + dt*veloPrev;
    elif (predictor_type == 2):
        dispPred = dispPrev + dt*(1.5*veloPrev-0.5*veloPrev2);
    else:
        print("Predictor type is invalid. \nValid options are 0, 1 and 2. ")

    #print("dispPred = %f \n" % dispPred)

    veloFluid = (dispPred-dispPredPrev)/dt;
    #veloFluid = 2.0*(dispPred-dispPredPrev)/dt - veloFluidPrev;

    acceFluid = (veloFluid-veloFluidPrev)/dt;

    ff = (1.0-alpha)*acceFluid + 2.0*xi*w*veloFluid ;

    # store force values
    fs   = -ff;

    # solid problem
    #
    # force on the solid at t_{n+1}
    fsnp1 = fs;
    disp = dispPrev

    for iter in range(5):

        velo = (disp - dispPrev)/dt;
        acce = (velo - veloPrev)/dt;

        resi = fsnp1 - alpha*acce - w*w*disp ;
        rNorm = abs(resi);

        if( rNorm < 1.0e-8 ):
            break;

        disp = disp + resi/Klocal;
        # iteration loop ended here

    disp = beta*disp + (1.0-beta)*dispPred;

    dispNum[ii] = disp;
    veloNum[ii] = velo;
    veloFluidNum[ii] = veloFluid;


    # store solution variables
    # solid problem
    dispPrev  = disp;
    veloPrev2 = veloPrev;
    veloPrev  = velo;
    accePrev  = acce;

    dispPredPrev = dispPred;

    veloFluidPrev = veloFluid;
    acceFluidPrev = acceFluid;



# solution with the monolithic scheme
dispMono, veloMono = solution_monolithic_BE(m, c, k, mr, timesteparray)


plt.figure(0)
plot(timesteparray, dispMono,'g', linewidth=4)
plot(timesteparray, dispNum, 'k--', linewidth=2)
plt.show()

plt.figure(1)
plot(timesteparray, veloFluidNum,'g', linewidth=4)
plot(timesteparray, veloNum, 'k--', linewidth=2)

plt.show()
#legend('Mono','Stag')
#axis([0.0 tt(end) -1.2 1.2])







