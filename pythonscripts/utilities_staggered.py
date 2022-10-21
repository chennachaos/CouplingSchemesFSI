import numpy as np
from numpy import linalg as LA


###################################################
#
# Plot of Spectral radius Vs time step
#
# Scheme: Staggered with force predictor
# Fluid time integration: backward Euler
# Solid time integration: backward Euler
#
###################################################

def spectralRadius_staggered_forcerelaxation_BE(m, c, k, mr, predictor_type, beta, logtimestep):
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

    alpha = mr/(1.0+mr);

    p1 = 1.0;
    p2 = 0.0;

    if (predictor_type == 2):
        p1 =  2.0;
        p2 = -1.0;

    Nt = np.size(logtimestep);
    
    specRad = np.zeros((Nt,1), dtype=float)
    A = np.zeros((4,4), dtype=float)

    # time loop
    for ii in range(Nt):
        dt = (10.0**logtimestep[ii])*T

        # amplification matrix

        Z = beta*(alpha-1.0) - (alpha+w*w*dt*dt)*(beta-1)

        A[0,0] = alpha;
        A[0,1] = alpha;
        A[0,2] = p1;
        A[0,3] = p2;

        A[1,0] = -w*w*dt*dt;
        A[1,1] = alpha;
        A[1,2] = p1;
        A[1,3] = p2;
        
        A[2,0] = w*w*dt*dt*beta*(1-alpha); 
        A[2,1] = w*w*dt*dt*beta*(1.0-alpha);
        A[2,2] = p1*Z;
        A[2,3] = p2*Z;
        
        A[3,0] = 0.0;
        A[3,1] = 0.0;
        A[3,2] = alpha+w*w*dt*dt;
        A[3,3] = 0.0;


        fact = 1.0/(alpha+w*w*dt*dt)
        A = np.multiply(A, fact)
        
        evals, evecs = LA.eig(A)

        specRad[ii] = max(abs(evals))
    # time loop ends here

    return  specRad




###################################################
#
# Plot of Spectral radius Vs time step
#
# Scheme: Staggered with force predictor
# Fluid time integration: backward Euler
# Solid time integration: backward Euler
#
###################################################

def spectralRadius_staggered_displcementrelaxation_BE(m, c, k, mr, predictor_type, beta, logtimestep):
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

    alpha = mr/(1.0+mr);

    Nt = np.size(logtimestep);
    
    specRad = np.zeros((Nt,1), dtype=float)
    A = np.zeros((4,4), dtype=float)
    if(predictor_type > 1):
        A = np.zeros((5,5), dtype=float)

    # time loop
    for ii in range(Nt):
        dt = (10.0**logtimestep[ii])*T

        # amplification matrix

        fact  = w*w*dt*dt
        denom = alpha+w*w*dt*dt

        if (predictor_type == 0):
            A[0,0] = (beta*(2*alpha-1.0) + fact*(1-beta))/denom;
            A[0,1] = (alpha*beta)/denom;
            A[0,2] = (beta*(1.0-alpha))/denom;
            A[0,3] = (beta*(1.0-alpha))/denom;
            
            A[1,0] = beta*(alpha-fact-1.0)/denom;
            A[1,1] = alpha*beta/denom;
            A[1,2] = beta*(1.0-alpha)/denom;
            A[1,3] = beta*(1.0-alpha)/denom;
            
            A[2,0] =  1.0; 
            A[2,1] =  0.0;
            A[2,2] =  0.0;
            A[2,3] = -1.0;
            
            A[3,0] = 1.0;
            A[3,1] = 0.0;
            A[3,2] = 0.0;
            A[3,3] = 0.0;
        elif (predictor_type == 1):
            A[0,0] = (beta*(2*alpha-1.0) + fact*(1-beta))/denom;
            A[0,1] = (beta*(2*alpha-1.0) + fact*(1-beta))/denom;
            A[0,2] = (beta*(1.0-alpha))/denom;
            A[0,3] = (beta*(1.0-alpha))/denom;
            
            A[1,0] = beta*(alpha-fact-1.0)/denom;
            A[1,1] = (beta*(2*alpha-1.0) + fact*(1-beta))/denom;
            A[1,2] = beta*(1.0-alpha)/denom;
            A[1,3] = beta*(1.0-alpha)/denom;
            
            A[2,0] =  1.0; 
            A[2,1] =  1.0;
            A[2,2] =  0.0;
            A[2,3] = -1.0;
            
            A[3,0] = 1.0;
            A[3,1] = 1.0;
            A[3,2] = 0.0;
            A[3,3] = 0.0;
        elif (predictor_type == 2):
            A[0,0] = (beta*(2*alpha-1.0) + fact*(1-beta))/denom;
            A[0,1] = (beta*(5*alpha-3.0) + fact*3.0*(1-beta))/denom/2.0;
            A[0,2] = (beta*(1.0-alpha))/denom;
            A[0,3] = (beta*(1.0-alpha)+fact*(1-beta))/denom/2.0;
            A[0,4] = (beta*(1.0-alpha))/denom;
            
            A[1,0] = beta*(alpha-fact-1.0)/denom;
            A[1,1] = (beta*(5*alpha-3.0) + fact*3.0*(1-beta))/denom/2.0;
            A[1,2] = beta*(1.0-alpha)/denom;
            A[1,3] = (beta*(1.0-alpha)+fact*(1-beta))/denom/2.0;
            A[1,4] = beta*(1.0-alpha)/denom;
            
            A[2,0] =  1.0;
            A[2,1] =  1.5;
            A[2,2] =  0.0;
            A[2,3] = -0.5;
            A[2,4] = -1.0;
            
            A[3,0] =  0.0;
            A[3,1] =  1.0;
            A[3,2] =  0.0;
            A[3,3] =  0.0;
            A[3,4] =  0.0;

            A[4,0] =  1.0;
            A[4,1] =  1.5;
            A[4,2] =  0.0;
            A[4,3] = -0.5;
            A[4,4] =  0.0;

        evals, evecs = LA.eig(A)

        specRad[ii] = max(abs(evals))
    # time loop ends here

    return  specRad


###################################################
#
# staggered scheme with backward-Euler scheme
# force predictor
# solid solver first
#
# Step 1: Predict force on the solid
# Step 2: Solve the solid problem
# Step 3: Update the fluid mesh
# Step 4: Solve the fluid problem
# Step 5: Relax the force on the solid
# Step 6: Go to next time step
#
###################################################


def solution_staggered_forcerelaxation_BE(m, c, k, mr, predictor_type, beta, timesteparray):

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

    ## parameters 

    alpha = mr/(1.0+mr);              # auxiliary parameter

    ##
    # solution phase starts from here

    dispExct = np.zeros((Nt,1), dtype=float)
    dispNum  = np.zeros((Nt,1), dtype=float)
    veloNum  = np.zeros((Nt,1), dtype=float)


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

    ff = (1.0-alpha)*acceFluid + 2.0*xi*w*veloFluid ;


    fsn2 = fsn1;
    fsn1 = fsn;
    fsn  = fs;

    fs = -beta*ff + (1.0-beta)*fsP;

    dispPrev = disp;
    veloPrev = velo;
    accePrev = acce;

    veloFluidPrev = veloFluid;
    acceFluidPrev = acceFluid;


    # store the solutions at t=0
    dispNum[0] = d0;
    veloNum[0] = v0;

    Klocal = alpha/dt/dt + w*w;


    for ii in range(1,Nt):

        # solid problem
        #
        # force predictor
        fsP = q1*fs + q2*fsn + q3*fsn1 + q4*fsn2;

        # force on the solid at t_{n+1}
        fsnp1 = fsP;

        for iter in range(5):

            velo = (disp - dispPrev)/dt;
            acce = (velo - veloPrev)/dt;

            resi = fsnp1 - alpha*acce - w*w*disp ;
            rNorm = abs(resi);

            #print(' rNorm : %5d ...  %12.6E \n' % (iter, rNorm) );

            if( rNorm < 1.0e-8 ):
                break;

            disp = disp + resi/Klocal;
        # iteration loop ended here

        dispNum[ii] = disp;
        veloNum[ii] = velo;

        # fluid problem

        veloFluid = velo;

        acceFluid = (veloFluid-veloFluidPrev)/dt;

        ff = (1.0-alpha)*acceFluid + 2.0*xi*w*veloFluid ;

        # store force values
        fsn2 = fsn1;
        fsn1 = fsn;
        fsn  = fs;

        fs = -beta*ff + (1.0-beta)*fsP;            # force relaxation


        # store solution variables
        # solid problem
        dispPrev = disp;
        veloPrev = velo;
        accePrev = acce;

        veloFluidPrev = veloFluid;
        acceFluidPrev = acceFluid;

    # time loop ends here
    return dispNum, veloNum




###################################################
#
# staggered scheme with backward-Euler scheme
# displacement predictor
# solid solver first
#
# Step 1: Calculate an explicit predictor of the structural interface displacement at the new time level
# Step 2: Compute fluid velocity at the interface to serve as Ditichlet BC
# Step 3: Update the mesh displacement
# Step 4: Solve fluid equations for fluid velocity and pressure
# Step 5: Obtain fluid boundary traction along the interface
# Step 6: Solve the structural field for the new displacements d n+1 under consideration of new fluid load
# Step 7: Proceed to next time step
#
###################################################


def solution_staggered_displacementrelaxation_BE(m, c, k, mr, predictor_type, beta, timesteparray):

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

    ## parameters 
    alpha = mr/(1.0+mr);              # auxiliary parameter

    ##
    # solution phase starts from here

    dispNum  = np.zeros((Nt,1), dtype=float)
    veloNum  = np.zeros((Nt,1), dtype=float)

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

        #print("veloFluid = %f \n" % veloFluid)
        #print("acceFluid = %f \n" % acceFluid)
        #print("fs = %f \n" % fs)

        # solid problem
        #
        # force on the solid at t_{n+1}
        fsnp1 = fs;
        disp = dispPrev

        for iter in range(5):

            #print("disp ....  %f \t %f \n" % (disp, dispPrev))
            #print("velo ....  %f \t %f \n" % (velo, veloPrev))

            velo = (disp - dispPrev)/dt;
            acce = (velo - veloPrev)/dt;
            #acce = (disp - dispPrev)/dt/dt - veloPrev/dt;

            #print("acce ....  %f \n" % acce)

            resi = fsnp1 - alpha*acce - w*w*disp ;
            rNorm = abs(resi);

            #print("resi ....  %f \n" % resi)

            #print(' rNorm : %5d ...  %12.6E \n' % (iter, rNorm) );

            if( rNorm < 1.0e-8 ):
                break;

            disp = disp + resi/Klocal;
            #print("disp = %f \n" % disp)
        # iteration loop ended here

        #print("disp = %f \n" % disp)
        #print("velo = %f \n" % velo)

        disp = beta*disp + (1.0-beta)*dispPred;

        dispNum[ii] = disp;
        veloNum[ii] = velo;


        # store solution variables
        # solid problem
        dispPrev  = disp;
        veloPrev2 = veloPrev;
        veloPrev  = velo;
        accePrev  = acce;

        dispPredPrev = dispPred;

        veloFluidPrev = veloFluid;
        acceFluidPrev = acceFluid;

    # time loop ends here
    return dispNum, veloNum






###################################################
#
# staggered scheme with backward-Euler scheme
# force predictor
# solid solver first
#
###################################################


def solution_staggered_combinedrelaxation_BE(m, c, k, mr, predictor_type, beta, timesteparray):

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

    B1 = d0;
    B2 = (v0+xi*w*d0)/wd;

    B = np.sqrt(B1*B1+B2*B2);
    phi = np.arctan(B2/B1);


    dt = timesteparray[1] - timesteparray[0];

    Nt = np.size(timesteparray);

    ## parameters 

    alpha = mr/(1.0+mr);              # auxiliary parameter

    ##
    # solution phase starts from here

    dispExct = np.zeros((Nt,1), dtype=float)
    dispNum  = np.zeros((Nt,1), dtype=float)
    veloNum  = np.zeros((Nt,1), dtype=float)

    #uMono = soln_mono_BE(m, c, k, d0, v0, dt);
    uMono = veloNum


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

    ff = (1.0-alpha)*acceFluid + 2.0*xi*w*veloFluid ;


    fsn2 = fsn1;
    fsn1 = fsn;
    fsn  = fs;

    fs = -beta*ff + (1.0-beta)*fsP;

    dispPrev = disp;
    veloPrev = velo;
    accePrev = acce;

    veloFluidPrev = veloFluid;
    acceFluidPrev = acceFluid;


    # store the solutions at t=0
    dispNum[0] = d0;
    veloNum[0] = v0;
    dispExct[0] = np.exp(-xi*w*timesteparray[0])*B*np.cos(wd*timesteparray[0]-phi);


    Klocal = alpha/dt/dt + w*w;


    for ii in range(1,Nt):

        dispExct[ii] = np.exp(-xi*w*timesteparray[ii])*B*np.cos(wd*timesteparray[ii]-phi);

        # solid problem
        #
        # force predictor
        fsP = q1*fs + q2*fsn + q3*fsn1 + q4*fsn2;

        # force on the solid at t_{n+1}
        fsnp1 = fsP;

        for iter in range(5):

            velo = (disp - dispPrev)/dt;
            acce = (velo - veloPrev)/dt;

            resi = fsnp1 - alpha*acce - w*w*disp ;
            rNorm = abs(resi);

            #print(' rNorm : %5d ...  %12.6E \n' % (iter, rNorm) );

            if( rNorm < 1.0e-8 ):
                break;

            disp = disp + resi/Klocal;
        # iteration loop ended here

        dispNum[ii] = disp;
        veloNum[ii] = velo;

        # fluid problem
        #veloFluid = velo;

        #veloFluid = beta*veloFluidPrev + (1.0-beta)*velo;            # velocity relaxation
        veloFluid = beta*velo + (1.0-beta)*veloFluidPrev;            # velocity relaxation


        acceFluid = (veloFluid-veloFluidPrev)/dt;

        ff = (1.0-alpha)*acceFluid + 2.0*xi*w*veloFluid ;

        # store force values
        fsn2 = fsn1;
        fsn1 = fsn;
        fsn  = fs;

        fs = -beta*ff + (1.0-beta)*fsP;            # force relaxation


        # store solution variables
        # solid problem
        dispPrev = disp;
        veloPrev = velo;
        accePrev = acce;

        veloFluidPrev = veloFluid;
        acceFluidPrev = acceFluid;

    # time loop ends here
    return dispNum, veloNum


