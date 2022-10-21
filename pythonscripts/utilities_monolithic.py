import numpy as np
from numpy import linalg as LA


###################################################
#
# Plot of Spectral radius Vs time step
#
# Scheme: Monolithic
# Fluid time integration: backward Euler
# Solid time integration: backward Euler
#
###################################################

def spectralRadius_monolithic_BE(m, c, k, mr, logtimestep):
    ## Input parameters
    #
    #m  = 1.0;                        # total mass, m=ms+mf
    #k  = 4.0*np.pi**2;               # stiffness
    #c  = 0.0;                        # damping coefficient
    #mr = 10.0;                       # mass ratio

    ## Intermediate (auxiliary) parameters
    #
    w  = np.sqrt(k/m);               # (circular) natural frequency
    xi = c/(2.0*np.sqrt(k*m));       # damping ratio
    wd = w*np.sqrt(1.0-xi*xi);       # damped natural frequency
    T  = 2.0*np.pi/w;                # time period

    #ms = m*(mr/(1.0+mr));            # mass for the solid (spring-mass) system
    #mf = m-ms;                       # mass for the fluid (damper-mass) system
    a = mr/(1.0+mr);                  # parameter alpha (=ms/m=mr/(1+mr))
    
    Nt = np.size(logtimestep);

    specRad = np.zeros((Nt,1), dtype=float)

    # time loop
    for ii in range(Nt):
        dt = (10.0**logtimestep[ii])*T

        # amplification matrix
    
        A = [[1.0+2.0*xi*w*dt,                1.0,                        0.0],
             [-w*w*dt*dt,                     1.0,                        0.0],
             [ w*w*dt*dt*(1.0-a+2.0*xi*w*dt), w*dt*(w*dt*(1.0-a)-2*xi*a), 0.0] ]
    
        #A[0,0] = 1.0+2.0*xi*w*dt;              A[0,1] = 1.0;                            A[0,2] = 0.0;
        #A[1,0] = -w*w*dt*dt;                   A[1,1] = 1.0;                            A[1,2] = 0.0;
        #A[2,0] = w*w*dt*dt*(1-a+2*xi*w*dt);    A[2,1] = w*dt*(w*dt*(1.0-a)-2*xi*a);     A[2,2] = 0.0;

        fact = 1.0/(1.0+2.0*xi*w*dt+w*w*dt*dt)
        A = np.multiply(A, fact)

        evals, evecs = LA.eig(A)
    
        specRad[ii] = max(abs(evals))
    # time loop ends here

    return  specRad



###################################################
#
# monolithic scheme with backward-Euler scheme
#
###################################################


def solution_monolithic_BE(m, c, k, mr, timesteparray):

    ## Input parameters
    #
    #m  = 1.0;                        # total mass, m=ms+mf
    #k  = 4.0*np.pi**2;               # stiffness
    #k  = 1.0;
    #c  = 0.0;                        # damping coefficient
    #mr = 10.0;                       # mass ratio

    d0 = 1.0;                        # initial displacement
    v0 = 0.0;                        # initial velocity
    a0 = (-c*v0 - k*d0)/m;           # initial acceleration

    ## Intermediate (auxiliary) parameters
    #
    w  = np.sqrt(k/m);               # (circular) natural frequency
    xi = c/(2.0*np.sqrt(k*m));       # damping ratio
    wd = w*np.sqrt(1.0-xi*xi);       # damped natural frequency
    T  = 2.0*np.pi/w;                # time period

    dt = timesteparray[1] - timesteparray[0];

    Nt = np.size(timesteparray);

    ##
    # solution phase starts from here

    dispNum  = np.zeros((Nt,1), dtype=float)
    veloNum  = np.zeros((Nt,1), dtype=float)


    # initialise variables used in the solution
    dispPrev = 0.0;
    veloPrev = 0.0;
    accePrev = 0.0;

    disp = d0;
    velo = v0;
    acce = a0;

    dispPrev = disp;
    veloPrev = velo;
    accePrev = acce;

    # store the solutions at t=0
    dispNum[0] = d0;
    veloNum[0] = v0;


    Klocal = 1.0/dt/dt + 2.0*xi*w/dt + w*w;


    for ii in range(1,Nt):

        # force on the solid at t_{n+1}
        fsnp1 = 0.0;

        for iter in range(5):

            velo = (disp - dispPrev)/dt;
            acce = (velo - veloPrev)/dt;

            resi = fsnp1 - acce - 2.0*xi*w*velo - w*w*disp ;
            rNorm = abs(resi);

            #print(' rNorm : %5d ...  %12.6E \n' % (iter, rNorm) );

            if( rNorm < 1.0e-8 ):
                break;

            disp = disp + resi/Klocal;
        # iteration loop ended here

        dispNum[ii] = disp;
        veloNum[ii] = velo;

        # store solution variables
        # solid problem
        dispPrev = disp;
        veloPrev = velo;
        accePrev = acce;

    # time loop ends here
    return dispNum, veloNum

