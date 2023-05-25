# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 14:30:46 2023

@author: Hilde
"""
"""
in monte carlo evaluation of gluonic action path integrals 
the gluon field is specified not by coordinates, but by link variuables
which are exponentials of the field A. to update a link variable 
we must multiply by the exponential of a random field, that is 
a random SU3 matrix M. 

We generate 50-100 random SU3 matricies at once, then sample from that 
to update our link variables. the only restriction on this is that the set be large
enough so that the products of the Ms cover the entire space of SU3, and that the inverse 
of each matric be in the set. 

second modification of the metropolis algorithm in QCD is that we update each link variable several 
times before moving on to the next variable in the sweep. this allows the link variable to come into statistical
equilibrium with its immediate neighbors. 

the part of the action that must be computed when updating a particular link variable
can be written
DeltaS = ReTr(U_mu(x)*Gamma_mu(x)), where Gamma is the sum of products of the link variables
computing the gammas is the most expensive part of the update, but it is only needed once for 
each set of successive updates of U. Usually we do about 10 hits of the metropolis algorithm before 
moving to the next link variable. 

Exercise: evaluate the gluonic path integrals using the metropolis algorithm. use the Wilson action then try the improved action
beta = beta hat /u0^4 = 5.5, this corresponds to a lattice spacinf og 0.25 fm
L/a = 8 points on a side giving lattice volume on the order of 2fm. (8x8x8x8 lattice) metropolis step size 
epsilon = 0.24, omit Ncor = 50 between measurements. compute averages of a x a and a x 2a. 
for wilson action, values should be 0.50, 0.26
for improved action, values should be 0.54, 0.28. 

"""
import numpy as np
import math as math
import time
from numpy import dot

start_time = time.time()

#function that takes the hermeitan conjugate of a square matrix
def dagger(M):
    M_dagger = np.conjugate(M.T)
    return M_dagger

#Generate 100 random SU3 matricies
def randomSU3(Mn):
    MS = np.zeros([200,3,3],dtype=complex)
    HS = np.zeros([200,3,3],dtype=complex)
    for i in range(Mn):
        #these loops generates a Hermitian matrix
        R = np.zeros([3,3],dtype=complex)
        for j in range(3):
            for k in range(3):
                n = np.random.uniform(-1,1) #we create a set of Hermitian matricies with elements random numbers between -1 and 1
                m = np.random.uniform(-1,1) 
                R[j,k] = complex(n,m) #complex matrix of random numbers from -1 to 1. 
                
        #we now want this created matrix to be Hermitian, so we take its adjoint and normalize
        H = (R.copy() + dagger(R.copy()))/2 #creation of the Hermitian matrix
        HS[i] = H.copy()
        U = np.zeros([3,3],dtype=complex)
        #Now that we have a Hermitian matrix, we can create a unitary matrix U = exp(iH) in terms of its Taylor series expansion

        for p in range(50): #terms of taylor expansion
            U += (((1j*eps)**p)/math.factorial(p))*np.linalg.matrix_power(H, p) 
        #This gives us a Unitary matrix, an element of U(3), we now need to make it an element of SU(3)
        #To do this, we need to make it's determinant 1, so we divide the matrix by the cube root of its determiniant to enforce this
        U = U/(np.linalg.det(U)**(1/3)) 
        MS[i] = U
        MS[i+Mn] = dagger(MS[i])
        
    return MS.copy()

    
#we want to update the link variable U with SU(3) matrix M and caculate the change in action
#we iterate through each 4 dimensional lattice site, pull one of our random SU(3) matricies and multiply it to the old U as a 
#proposed update to U. We calculate the change in action with that update, if it lowers the action or if we roll smaller that e^-dS
#we keep the proposed change. at that site. we do this 10 times per site. 
#Inputs: U - array of link variables for each site represented by a SU(3) matrix. there is a link variable for every site, so U is written as U[x,y,z,t,mu,n,m]
#where the position of the link variable is given by the first 5 indicies, and n and m tell us the matrix element of U. 
#Inputs: M - array or random SU(3) matricies that we pull from to update the link variables. 
#Outputs: U_new - updated set of link variables.
def Update(U,M):
    for x in range(N):
        for y in range(N):
            for z in range(N):
                for t in range(N):
                    for mu in range(4): # we go to each lattice site and iterate through mu
                        G = Gamma(U,x,y,z,t,mu)
                        R = Omega(U,x,y,z,t,mu)
                        
                        for h in range(hits):
                            r = np.random.randint(0,2*Mn) 
                            U_loc = U[x,y,z,t,mu].copy() #storing the single 3x3 for this specific location as U_loc
                            U_update = M[r].copy()@U[x,y,z,t,mu].copy() #then creating a proposed update for this locations link variable
                            
                            if improved == True:
                                dS = (-beta_imp/3)*((5/(3*u0**4)*np.real(np.trace((M[r].copy()@U[x,y,z,t,mu].copy() - U[x,y,z,t,mu].copy())@G.copy()))) - (1/(12*u0**6))*np.real(np.trace((M[r].copy()@U[x,y,z,t,mu].copy() - U[x,y,z,t,mu].copy())@R.copy())))
                                #dS = -beta_imp/(3)*((5/(3*u0**4)*np.real(np.trace(dot((dot(M[r],U[x,y,z,t,mu])-U[x,y,z,t,mu]),G)))-1/(12*u0**6)*np.real(np.trace(dot((dot(M[r],U[x,y,z,t,mu])-U[x,y,z,t,mu]),R))))) # change in the improved action
                            else:
                                dS = -(beta/3)*np.real(np.trace((M[r].copy()@U[x,y,z,t,mu].copy() - U[x,y,z,t,mu].copy())@G.copy()))
                            
                            if dS < 0 or np.random.uniform(0,1) < np.exp(-dS):
                                U[x,y,z,t,mu] = U_update.copy()
                            else:
                                U[x,y,z,t,mu] = U_loc.copy()


#calculate gamma 
#When we change a link variable in update, we need to get the "staples" or a product of the link variables the link we changed is connected to
#This means we need to iterate through the free index, nu, and calculate the sum of each possible path, forward and backward, that connects the updated link variable. 
#this means we look at when nu != mu, and assemble the staple as a product of links that would create the planquette when the updated link is included in the product, which it is in the action
#
#inputs: U - set of link operators. 
#        x,y,z,t - indicies designates the position on the lattice that we are calculating gamma for. 
#        ind - this is the mu that we are changing the link variable for.  

def Gamma(U,x,y,z,t,ind):
    gamma = 0
    mu_vec = np.zeros(4,dtype=int) #these are "4-vectors" representing the indicies of P_mu_nu and U_mu, which we need as we assemble our products of links
    nu_vec = np.zeros(4,dtype=int) # we want the mu one to only have one in the direction we updated in, then we iterate through nu
    
    
    #defining positions on the lattice in terms of steps of mu or nu. 
    loc = np.array([x,y,z,t]) #initial position, the origin of the link we are updating this time
    loc_mu = np.zeros(4,dtype=int) #position on the next lattice site in the mu direction
    loc_mu_nu = np.zeros(4,dtype=int) #position diagonal to the input position #this isnt used
    loc_nu = np.zeros(4,dtype=int) #position of next lattice site in the nu direction
    loc_n_nu = np.zeros(4,dtype=int) #position of the next lattice site in the negative nu direction
    loc_mu_n_nu = np.zeros(4,dtype=int) #position diagonal to the input posittion, in the positive mu and negative nu direction
    
    loc = np.array([x,y,z,t])
    mu_vec[ind] = 1 #mu is an input to this function, we only want to increment mu in one direction, so we have the input mu be one, and the rest be zero
    
    #defining loc_mu as + one spot in the mu direction
    #for i in range(4):
    loc_mu = (loc + mu_vec)%N 
        
    #now we iterate through nu and create products of every possible staple
    for nu in range(4):
        if nu != ind: #exclusive to the square operator, our next step in the surrounding paths will always be perpendicular to the current mu
            nu_vec[nu] = 1 #we make it so the nu we want incremented is the only active element of our nu vector
            
            #define loc_nu as +1 spot in the nu direction relative to input position
            loc_nu = (loc + nu_vec)%N
            
            #define the diagonal position in the positive mu and negative nu
            loc_mu_n_nu = (loc_mu - nu_vec)%N
            
            #define the position -1 spot in the nu direction
            loc_n_nu = (loc - nu_vec)%N
                
            nu_vec[nu] = 0 #reset nu for the next loop
            
            #we assemble a product of the staple positions we just defined going above and below our updated link
            gamma +=  U[loc_mu[0],loc_mu[1],loc_mu[2],loc_mu[3],nu]@dagger(U[loc_nu[0],loc_nu[1],loc_nu[2],loc_nu[3],ind])@dagger(U[loc[0],loc[1],loc[2],loc[3],nu]) #positive nu product
            gamma +=  dagger(U[loc_mu_n_nu[0],loc_mu_n_nu[1],loc_mu_n_nu[2],loc_mu_n_nu[3],nu])@dagger(U[loc_n_nu[0],loc_n_nu[1],loc_n_nu[2],loc_n_nu[3],ind])@U[loc_n_nu[0],loc_n_nu[1],loc_n_nu[2],loc_n_nu[3],nu] #negative nu product

    return gamma.copy()

#Gamma equivalent for rectangle operators in the improved action. looking at staples of ax2a or 2axa
#just rewrite this shit please
def Omega(U,x,y,z,t,mu):
    omega = 0 
    mu_vec = np.zeros(4,dtype=int)
    mu_vec[mu] = 1
    
    #defining location vectors
    loc = np.array([x,y,z,t],dtype=int)
    
    loc_mu = (loc + mu_vec.copy())%N
    loc_2mu = (loc + 2*mu_vec.copy())%N
    loc_mmu = (loc - mu_vec.copy())%N
    
    for nu in range(4):
        if nu != mu:
            nu_vec = np.zeros(4,dtype=int)
            nu_vec[nu] = 1
            
            loc_nu = (loc + nu_vec.copy())%N
            loc_2nu = (loc + 2*nu_vec.copy())%N
            loc_m2nu = (loc - 2*nu_vec.copy())%N
            loc_mnu = (loc - nu_vec.copy())%N
            
            loc_mu_nu = (loc + mu_vec.copy() + nu_vec.copy())%N
            loc_mmu_mnu = (loc - mu_vec.copy() - nu_vec.copy())%N
            loc_mmu_nu = (loc - mu_vec.copy() + nu_vec.copy())%N
            loc_mu_mnu = (loc + mu_vec.copy() - nu_vec.copy())%N
            loc_mu_m2nu = (loc + mu_vec.copy() - 2*nu_vec.copy())%N   
            loc_2mu_mnu = (loc + 2*mu_vec.copy() - nu_vec.copy())%N
            
            omega += U[loc_mu[0],loc_mu[1],loc_mu[2],loc_mu[3],nu]@U[loc_mu_nu[0],loc_mu_nu[1],loc_mu_nu[2],loc_mu_nu[3],nu]@dagger(U[loc_2nu[0],loc_2nu[1],loc_2nu[2],loc_2nu[3],mu])@dagger(U[loc_nu[0],loc_nu[1],loc_nu[2],loc_nu[3],nu])@dagger(U[x,y,z,t,nu]) \
                + U[loc_mu[0],loc_mu[1],loc_mu[2],loc_mu[3],nu]@dagger(U[loc_nu[0],loc_nu[1],loc_nu[2],loc_nu[3],mu])@dagger(U[loc_mmu_nu[0],loc_mmu_nu[1],loc_mmu_nu[2],loc_mmu_nu[3],mu])@dagger(U[loc_mmu[0],loc_mmu[1],loc_mmu[2],loc_mmu[3],nu])@U[loc_mmu[0],loc_mmu[1],loc_mmu[2],loc_mmu[3],mu] \
                    + U[loc_mu[0],loc_mu[1],loc_mu[2],loc_mu[3],mu]@U[loc_2mu[0],loc_2mu[1],loc_2mu[2],loc_2mu[3],nu]@dagger(U[loc_mu_nu[0],loc_mu_nu[1],loc_mu_nu[2],loc_mu_nu[3],mu])@dagger(U[loc_nu[0],loc_nu[1],loc_nu[2],loc_nu[3],mu])@dagger(U[x,y,z,t,nu]) \
                        + U[loc_mu[0],loc_mu[1],loc_mu[2],loc_mu[3],mu]@dagger(U[loc_2mu_mnu[0],loc_2mu_mnu[1],loc_2mu_mnu[2],loc_2mu_mnu[3],nu])@dagger(U[loc_mu_mnu[0],loc_mu_mnu[1],loc_mu_mnu[2],loc_mu_mnu[3],mu])@dagger(U[loc_mnu[0],loc_mnu[1],loc_mnu[2],loc_mnu[3],mu])@U[loc_mnu[0],loc_mnu[1],loc_mnu[2],loc_mnu[3],nu] \
                            + dagger(U[loc_mu_mnu[0],loc_mu_mnu[1],loc_mu_mnu[2],loc_mu_mnu[3],nu])@dagger(U[loc_mu_m2nu[0],loc_mu_m2nu[1],loc_mu_m2nu[2],loc_mu_m2nu[3],nu])@dagger(U[loc_m2nu[0],loc_m2nu[1],loc_m2nu[2],loc_m2nu[3],mu])@U[loc_m2nu[0],loc_m2nu[1],loc_m2nu[2],loc_m2nu[3],nu]@U[loc_mnu[0],loc_mnu[1],loc_mnu[2],loc_mnu[3],nu] \
                                + dagger(U[loc_mu_mnu[0],loc_mu_mnu[1],loc_mu_mnu[2],loc_mu_mnu[3],nu])@dagger(U[loc_mnu[0],loc_mnu[1],loc_mnu[2],loc_mnu[3],mu])@dagger(U[loc_mmu_mnu[0],loc_mmu_mnu[1],loc_mmu_mnu[2],loc_mmu_mnu[3],mu])@U[loc_mmu_mnu[0],loc_mmu_mnu[1],loc_mmu_mnu[2],loc_mmu_mnu[3],nu]@U[loc_mmu[0],loc_mmu[1],loc_mmu[2],loc_mmu[3],mu]
   
    return omega.copy()


#calculate Wilson a x a loop
#function computes the Wilson action of every possible axa loop for every point on the lattice, we dont go backwards like we did in gamma because 
#that backwards loop is the same as one of the forward loops at a different point on the lattice, the one starting -1 spot in the nu direction from the current spot
#WL_axa = 1/3 * ReTr(U_mu(r)*U_nu(r + a*mu)*U_mu_dagger(r + a*nu)*U_nu_dagger(r))
def Wilson_axa(U,x,y,z,t):
    WL = 0
    mu_vec = np.zeros(4,dtype=int)
    nu_vec = np.zeros(4,dtype=int)
    
    loc = np.array([x,y,z,t],dtype=int)
    loc_mu = np.zeros(4,dtype=int)
    loc_nu = np.zeros(4,dtype=int)
    
    for mu in range(4):
        mu_vec[mu] = 1
        loc_mu = (loc + mu_vec)%N
        
        for nu in range(mu):
            nu_vec[nu] = 1
            loc_nu = (loc + nu_vec)%N
            
            WL += np.trace(U[loc[0],loc[1],loc[2],loc[3],mu]@U[loc_mu[0],loc_mu[1],loc_mu[2],loc_mu[3],nu]@dagger(U[loc_nu[0],loc_nu[1],loc_nu[2],loc_nu[3],mu])@dagger(U[loc[0],loc[1],loc[2],loc[3],nu]))
            
            nu_vec[nu] = 0 #resetting our indicies
        mu_vec[mu] = 0
    
    return np.real(WL)/(3.*6.)
#notice giw we loop mu from 0 to 4, but then nu only from 0 to mu. This makes sure we calculate every axa loop from the given point 
#(x,y,z,t) only once. mu starts as the x direction, nothing is done. mu is then in the y direction, nu can only be in the x. that gives us one loop
#mu in the z direction allows nu to be in the x or y direction, giving 2 loops, mu in the t direction allows nu to be x,y,z direction, giving 3 loops. in total that is 6 axa loops, which is 
#the correct amount for any given point with no negative directions
#for example in 2 dimensions, from any given point on the lattice, there is only one possible loop. so mu could be 1 or 2, nu could also be 1 or 2, but we don't want to double count because mu = 1, nu = 2 is the same loop as mu = 2 nu = 1. thus we loop nu in a range from 0 to mu, so that we only count this loop once
#in 3D, there are 3 axa loops at any point, mu in the y nu in the x (mu = 1 nu = 0) mu in the z nu in the x (mu = 2 nu = 0) or mu in the z nu in the y (mu=2,nu = 1). this again is represented by mu ranging from 0 to 3, then nu ranging from 0 to mu, so that we only get each possible loop once. keep in mind that mu and nu can be swapped and we get the same loop. 
#finally in 4 dimensions, we have mu in the y nu in the x, mu in the z nu in the x or y, mu in the t nu in the x y or z, and that covers every possible axa loop on our lattice. 
#looping like this also prevents nu = mu, which cannot be possible. 

#calculate Wilson 2a x a loop
def Wilson_ax2a(U,x,y,z,t):
    W = 0
    mu_vec = np.zeros(4,dtype=int)
    nu_vec = np.zeros(4,dtype=int)
    
    #we need these specific ones based on the product of link variables that are needed to calculate the action for an ax2a loop.
    loc = np.array([x,y,z,t],dtype=int) #origin for this specific action calculation
    loc_mu = np.zeros(4,dtype=int) # +1 in the mu direction
    loc_nu = np.zeros(4,dtype=int) # +1 in the nu direction
    loc_mu_nu = np.zeros(4,dtype=int) # +1 in both directions
    loc_2nu = np.zeros(4,dtype=int) # +2 in the nu direction
    
    #we iterate through mu and nu to get every possible ax2a loop from the origin input to the function
    for mu in range(4):
        mu_vec[mu] = 1 
        
        loc_mu = (loc + mu_vec)%N #trying out vector addition here instead of writing out each dimension manually or using a loop
        
        for nu in range(0,mu):
            nu_vec[nu] = 1
            
            loc_nu = (loc + nu_vec)%N
            loc_mu_nu = (loc_mu + nu_vec)%N
            loc_2nu = (loc_nu + nu_vec)%N
            
            W += np.trace(U[loc[0],loc[1],loc[2],loc[3],mu]@U[loc_mu[0],loc_mu[1],loc_mu[2],loc_mu[3],nu]@U[loc_mu_nu[0],loc_mu_nu[1],loc_mu_nu[2],loc_mu_nu[3],nu]@dagger(U[loc_2nu[0],loc_2nu[1],loc_2nu[2],loc_2nu[3],mu])@dagger(U[loc_nu[0],loc_nu[1],loc_nu[2],loc_nu[3],nu])@dagger(U[loc[0],loc[1],loc[2],loc[3],nu]))
            
            
            nu_vec[nu] = 0
        mu_vec[mu] = 0
        
    return np.real(W)/(3*6)

#for the ax2a, we have the exact same layout for the mu nu loop, but the shape of the loop is different

#main function that does the monte carlo process
#constants
Ncor = 50 #sweeps between MC measurements thermalize 5*Ncor times to make a good starting config
Ncf = 10 #number of configurations
eps = 0.24
N = 8
beta = 5.5
beta_imp = 1.719
u0 = 0.797
a = 0.25
Mn = 100 #number of random matricies
hits = 10 #number of changes to U at each site.
improved = True

M = randomSU3(Mn) #initializes M as the array that holds all of our random matricies
WL = np.zeros((Ncf)) #array to hold averaged wilson loop values. there are Ncf values to hold
WL_R = np.zeros((Ncf))
U = np.zeros((N,N,N,N,4,3,3),dtype=complex) #Array to hold link variables. first four indicies are the size of the lattice, the 4 is for each dimension, and the 3,3 tells us that each link variable is a 3x3 matrix
for i in range(N):
    for j in range(N):
        for k in range(N):
            for l in range(N):
                for mu in range(4):
                    for n in range(3):
                        U[i,j,k,l,mu,n,n] = 1 #this initializes our link variables as "1" or the identity
                        
for p in range(0,2*Ncor):
    Update(U,M)
print('Lattice Thermalized: %.4f Seconds'%(time.time() - start_time))

print('\n Wilson loop value for axa and ax2a \n')
for a in range(0,Ncf):
    for j in range(0,Ncor): #imposes statistical independence between measurements   
        Update(U,M)
    
    WL[a] = 0 #initialize the wilsons specific to this configuration
    WL_R[a] = 0
    
    #now we iterate through each lattice site and calculate the Wilson there, and add it to the total to be averaged once we've moved through the entire lattice
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    WL[a] += Wilson_axa(U, i, j, k, l)
                    WL_R[a] += Wilson_ax2a(U, i, j, k, l)
    
    WL[a] = WL[a]/(N**4) #averages our wilsons over each site on the lattice
    WL_R[a] = WL_R[a]/(N**4)
    print(a+1,':',' ', WL[a],'    ', WL_R[a])
    
    
avg_WL = 0
avg_WLR = 0
avg_WLSQ = 0
avg_WLRSQ = 0

for a in range(Ncf):
    avg_WL += WL[a]
    avg_WLR += WL_R[a]
    avg_WLSQ += WL[a]**2
    avg_WLRSQ += WL_R[a]**2

avg_WL = avg_WL/Ncf
avg_WLR /= Ncf
avg_WLSQ = avg_WLSQ/Ncf
avg_WLRSQ /= Ncf

err_avg_WL = (abs(avg_WLSQ - avg_WL**2)/Ncf)**(1/2)
err_avg_WLR = (abs(avg_WLRSQ - avg_WLR**2)/Ncf)**(1/2)

print('\n')

print('Monte Carlo average of axa Wilson loop with Wilson Action: %.4f +/- %.4f'%(avg_WL,err_avg_WL))
print('Monte Carlo average of ax2a Wilson loop with Wilson Action: %.4f +/- %.4f'%(avg_WLR,err_avg_WLR))    

print('%.4f Seconds'%(time.time() - start_time))

#notes
#Main rule for indicies of link variables, non-daggered are indexed based on where they come from, daggered U is indexed based on where it's going.
#Wilson loop functions are independent of the action we use to update the link variables, the axa loop and ax2a loop are things we want to calculate after we develop the lattice. 
#for the updated action, and gamma more specifically, we all possible paths on the lattice based on what is in the action. for the simple Wilson action, the only operator in it is a P_mu_nu operator,
#meaning we just need every possible square path including the link variable we are currently updating. 
#for the improved action, there are rectangle operators, which are 2axa and ax2a operators, meaning we need every possible square, ax2a and 2axa rectangle possible involving the link variable we are updating. This is only calculated once for each lattice site before it is hit 10 times
#for each lattice site, there are 8 directions you could go, and the fifth index on U represents that, as the link variables are the things connecting lattice sites.
#MC average: Thermalize the system by calling update 100 times, update iterates through the whole lattice and updates each link variable 10 times. 
#Then after thermalization, we start calculating wilson loop values with their respective functions. We create Ncf configurations.,
#then every Ncor updates we calculate a square and rectangle wilson loop value for each lattice site. we average all quare and rectangle values for that iteration, and that gives the wilson loop value for that specific configuration
#this is done Ncf times, and we then average these values over each configuration as well. 

#The naming of mu and nu is pure convention, switching mu and nu give the same loop but backwards, so depending on what we call forwards, we need to stick to it. for example if at some 
#x,y,z,t I call the mu = 1 nu = 3 loop forward, I need to do that for all spots on the lattice. this is inconvenient for looping through those indicies though, so we call the forward direction the one where mu > nu always, so that 
#we can convineinetly loop through the points and indicies. for example, at some x,y,z,t: the axa loop formed by link variables defined by mu = 2 nu = 1 is the exact same loop as the one at x,y+a,z,t with mu = 1 nu = 2. so all of our loops are covered 
#if we use the mu>nu convention for the definition of our axa loops. we will not couble count loops if we do mu>nu all the way, and its more convienient for iterating through the indicies themselves as we calculate each possible loop from a single point. 




