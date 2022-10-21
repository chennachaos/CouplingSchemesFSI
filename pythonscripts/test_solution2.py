from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from utilities_monolithic import *
from utilities_staggered import *


## Input parameters
#
m  = 1.0;                        # total mass, m=ms+mf
k  = 4.0*np.pi**2;               # stiffness
c  = 0.0;                        # damping coefficient

d0 = 1.0;                        # initial displacement
v0 = 0.0;                        # initial velocity
a0 = (-c*v0 - k*d0)/m;           # initial acceleration

w  = np.sqrt(k/m);               # (circular) natural frequency
xi = c/(2.0*np.sqrt(k*m));       # damping ratio
wd = w*np.sqrt(1.0-xi*xi);       # damped natural frequency
T  = 2.0*np.pi/w;                # time period

dt = T/200.0;

timesteparray = np.arange(0.0, 5*T+dt, dt)

mr = 2.9;                       # mass ratio

predictor_type = 1

A = np.zeros((4,4), dtype=float)
if (predictor_type == 1):
    A = np.zeros((5,5), dtype=float)

alpha = mr/(1.0+mr);
beta  = alpha;
beta  = 1.0;

fact  = w*w*dt*dt
denom = alpha+w*w*dt*dt

Klocal = alpha/dt/dt + w*w;

if (predictor_type == 1):
    #A[0,0] = (alpha*beta + alpha - beta + fact*(1-beta))/denom;
    #A[0,1] = (alpha + fact*(1-beta))/denom;
    #A[0,2] = (beta*(1.0-alpha))/denom;
    #
    #A[1,0] = (alpha-fact-1.0)/denom;
    #A[1,1] = alpha/denom;
    #A[1,2] = (1.0-alpha)/denom;
    #
    #A[2,0] = 1.0; 
    #A[2,1] = 1.0;
    #A[2,2] = 0.0;

    #A[0,0] = (alpha*beta + alpha - beta + fact*(1-beta))/denom;
    #A[0,1] = (alpha + fact*(1-beta))/denom;
    #A[0,2] = 0.0;
    #A[0,3] = (beta*(1.0-alpha))/denom;
    #
    #A[1,0] = (alpha-fact-1.0)/denom;
    #A[1,1] = alpha/denom;
    #A[1,2] = 0.0;
    #A[1,3] = (1.0-alpha)/denom;
    #
    #A[2,0] = 0.0; 
    #A[2,1] = 1.0;
    #A[2,2] = 0.0;
    #A[2,3] = 0.0;
    #
    #A[3,0] = 1.0;
    #A[3,1] = 1.0;
    #A[3,2] = 0.0;
    #A[3,3] = 0.0;

    A[0,0] = (2.0*alpha-1.0)/denom;
    A[0,1] = (2.0*alpha-1.0)/denom;
    A[0,2] = (1.0-alpha)/denom;
    A[0,3] = 0.0;
    A[0,4] = (1.0-alpha)/denom;
    
    A[1,0] = (alpha-fact-1.0)/denom;
    A[1,1] = (2.0*alpha-1.0)/denom;
    A[1,2] = (1.0-alpha)/denom;
    A[1,3] = 0.0;
    A[1,4] = (1.0-alpha)/denom;
    
    A[2,0] =  1.0;
    A[2,1] =  1.0;
    A[2,2] =  0.0;
    A[2,3] =  0.0;
    A[2,4] = -1.0;
    
    A[3,0] = 0.0;
    A[3,1] = 1.0;
    A[3,2] = 0.0;
    A[3,3] = 0.0;
    A[3,4] = 0.0;

    A[4,0] = 1.0;
    A[4,1] = 1.0;
    A[4,2] = 0.0;
    A[4,3] = 0.0;
    A[4,4] = 0.0;
elif (predictor_type == 2):
    A[0,0] = (alpha*beta + alpha - beta + fact*(1-beta))/denom;
    A[0,1] = (3.0*alpha - beta + 3.0*fact*(1-beta))/denom/2.0;
    A[0,2] = (-alpha + beta + fact*(beta-1.0))/denom/2.0;
    A[0,3] = beta*(1.0-alpha)/denom;

    A[1,0] = (alpha-fact-1.0)/denom;
    A[1,1] = (3.0*alpha-1.0)/denom/2.0;
    A[1,2] = (1.0-alpha)/denom/2.0;
    A[1,3] = (1.0-alpha)/denom;
    
    A[2,0] = 0.0; 
    A[2,1] = 1.0;
    A[2,2] = 0.0;
    A[2,3] = 0.0;
    
    A[3,0] =  1.0;
    A[3,1] =  1.5;
    A[3,2] = -0.5;
    A[3,3] =  0.0;

print(A)

Nt = np.size(timesteparray);
    
dispNum = np.zeros((Nt,1), dtype=float)


dispPrev = d0;
veloPrev = v0;
accePrev = a0;
veloPrev2 = 0.0;

dispPred  = d0;
dispPredPrev = d0;

veloFluidPrev = 0.0;

#solnPrev = [dispPrev, veloPrev*dt, veloPrev2*dt, dispPredPrev]
#solnPrev = [dispPrev, veloPrev*dt, dispPredPrev]

solnPrev = [dispPrev, veloPrev*dt, veloFluidPrev*dt, veloPrev2*dt, dispPredPrev]

soln = deepcopy(solnPrev)

dispNum[0] = d0

# time loop
for ii in range(1,Nt):

    soln = A.dot(solnPrev)
    #print("\n\n\n")
    #print(soln)
    #disp     = soln[0]
    #velo     = soln[1]/dt
    #veloPrev = soln[2]/dt
    #dispPred = soln[3]

    disp     = soln[0]
    velo     = soln[1]/dt
    dispPred = soln[2]

    #print("dispPred = %f \n" % dispPred)veloPrev*dt, 

    veloFluid = (dispPred-dispPredPrev)/dt;
    #veloFluid = 2.0*(dispPred-dispPredPrev)/dt - veloFluidPrev;

    acceFluid = (veloFluid-veloFluidPrev)/dt;

    ff = (1.0-alpha)*acceFluid + 2.0*xi*w*veloFluid ;

    # store force values
    fs   = -ff;

    #print("veloFluid = %f \n" % veloFluid)
    #print("acceFluid = %f \n" % acceFluid)
    #print("fs = %f \n" % fs)

    val = (soln[0]-solnPrev[0])*Klocal
    
    #print("rhs = %f \n " % val)
    
    #print("disp     = %f \n" % disp)
    #print("velo     = %f \n" % velo)

    dispNum[ii] = soln[0]
    solnPrev = 1.0*soln

    veloFluidPrev = veloFluid


# solution with the monolithic scheme
dispMono, veloMono = solution_monolithic_BE(m, c, k, mr, timesteparray)


plt.plot(timesteparray, dispMono,'g', linewidth=4)
#plt.plot(timesteparray, dispNumfP1,'k', linewidth=2)
plt.plot(timesteparray, dispNum,'b--')
plt.xlabel(r"$\Delta t/T$", fontsize=14)
plt.ylabel(r"Spectral Radius ($\rho$)", fontsize=14)
plt.xlim([min(timesteparray), max(timesteparray)])
#plt.ylim([0.0, 1.2])
plt.grid()
plt.show()



#[[ 0.49926087  0.99852175  0.          0.49926087]
# [-0.50073913  0.99852175  0.          0.49926087]
# [ 0.          1.          0.          0.        ]
# [ 1.          1.          0.          0.        ]]
#
#
#
#
#[ 0.99852175 -0.00147825  0.          1.        ]
#
#
#
#
#[ 0.99630765 -0.0022141  -0.00147825  0.9970435 ]