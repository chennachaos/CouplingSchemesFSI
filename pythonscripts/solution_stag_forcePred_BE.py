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


## Input parameters
#
m  = 1.0;                        # total mass, m=ms+mf
k  = 4.0*np.pi**2;               # stiffness
k  = 1.0;
c  = 0.0;                        # damping coefficient

d0 = 1.0;                        # initial displacement
v0 = 0.0;                        # initial velocity
a0 = (-c*v0 - k*d0)/m;           # initial acceleration

mr = 10.0;                       # mass ratio

## Intermediate (auxiliary) parameters
#
w  = np.sqrt(k/m);               # (circular) natural frequency
xi = c/(2.0*np.sqrt(k*m));       # damping ratio
wd = w*np.sqrt(1.0-xi*xi);       # damped natural frequency
T  = 2.0*np.pi/w;                # time period

ms = m*(mr/(1.0+mr));            # mass for the solid (spring-mass) system
mf = m-ms;                       # mass for the fluid (damper-mass) system



B1 = d0;
B2 = (v0+xi*w*d0)/wd;

B = np.sqrt(B1*B1+B2*B2);
phi = np.arctan(B2/B1);


dt = T/500.0;

tt = np.arange(0.0, 5*T+dt, dt)

Nt = size(tt);


## parameters 

alpha = ms/(mf+ms);               # auxiliary parameter
beta  = alpha;                    # relaxation parameter


p3 = -beta;
p4 = (1.0-beta);


##
# solution phase starts from here

dispExct = np.zeros((Nt,1), dtype=float)
dispNum  = np.zeros((Nt,1), dtype=float)
veloNum  = np.zeros((Nt,1), dtype=float)

#uMono = soln_mono_BE(m, c, k, d0, v0, dt);
uMono = veloNum


predictor_type = 2

if (predictor_type == 1):
    q1 =  1.0;  q2 = 0.0;  q3 =  0.0;  q4 =  0.0;
elif (predictor_type == 2):
    q1 =  2.0;  q2 = -1.0;  q3 =  0.0;  q4 =  0.0;
elif (predictor_type == 3):
    q1 =  3.0;  q2 = -3.0;  q3 =  1.0;  q4 = 0.0;
elif (predictor_type == 4):
    q1 =  4.0;  q2 = -6.0;  q3 =  4.0;  q4 = -1.0;
else:
    print("Predictor type is invalid. \nValid options are 1, 2, 3 and 4. ")



# initialise variables used in the solution
dispPrev = 0.0;
veloPrev = 0.0;
accePrev = 0.0;

veloFluidPrev = 0.0;
acceFluidPrev = 0.0;

fs   = 0.0;
fsn  = 0.0;
fsn1 = 0.0;
fsn2 = 0.0;
fsP  = 0.0;

disp = d0;
velo = v0;
acce = a0;

veloFluid = velo;

acceFluid = (veloFluid-veloFluidPrev)/dt ;
    
ff = mf*acceFluid + c*veloFluid ;


fsn2 = fsn1;
fsn1 = fsn;
fsn  = fs;

fs = p3*ff + p4*fsP;

dispPrev = disp;
veloPrev = velo;
accePrev = acce;

veloFluidPrev = veloFluid;
acceFluidPrev = acceFluid;


# store the solutions at t=0
dispNum[0] = d0;
veloNum[0] = v0;
dispExct[0] = np.exp(-xi*w*tt[0])*B*np.cos(wd*tt[0]-phi);


Klocal = ms/dt/dt + k; 


for ii in range(1,Nt):
    
    dispExct[ii] = np.exp(-xi*w*tt[ii])*B*np.cos(wd*tt[ii]-phi);

    # solid problem
    #
    # force predictor
    fsP = q1*fs + q2*fsn + q3*fsn1 + q4*fsn2;

    # force on the solid at t_{n+1}
    fsnp1 = fsP;

    for iter in range(5):

        velo = (disp - dispPrev)/dt;
        acce = (velo - veloPrev)/dt;

        resi = fsnp1 - ms*acce - k*disp ;
        rNorm = abs(resi);

        print(' rNorm : %5d ...  %12.6E \n' % (iter, rNorm) );
        
        if( rNorm < 1.0e-8 ):
            break;

        disp = disp + resi/Klocal;
    # iteration loop ended here

    dispNum[ii] = disp;
    veloNum[ii] = velo;

    # fluid problem

    veloFluid = velo;

    acceFluid = (veloFluid-veloFluidPrev)/dt;

    ff = mf*acceFluid + c*veloFluid ;

    # store force values
    fsn2 = fsn1;
    fsn1 = fsn;
    fsn  = fs;

    fs = p3*ff + p4*fsP;            # force relaxation


    # store solution variables
    # solid problem
    dispPrev = disp;
    veloPrev = velo;
    accePrev = acce;

    veloFluidPrev = veloFluid;
    acceFluidPrev = acceFluid;
    
# time loop ends here


plot(tt, dispExct,'k')
plot(tt, dispNum, 'b')

plt.show()
#legend('Mono','Stag')
#axis([0.0 tt(end) -1.2 1.2])







