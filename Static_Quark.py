import numpy as np
import scipy as sp
import math as math
import time
from numpy import dot
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import itertools
from SimpleLattice import Lattice #importing the lattice class from my other file that creates and thermalizes it. 

start_time = time.time()

#function that takes the hermeitan conjugate of a square matrix
def dagger(M):
    M_dagger = np.conjugate(M.T)
    return M_dagger


#############
    
    

#function that shuffles the measurement configurations for W, the array for the measurement of the rxa wilson loop.
#we pick a random integer between 0 and the number of configurations, and makes a new W matrix that holds the a position config in the ith spot, meaning we pick a random configuration, and put it in the first spot, then another randomly picked configuration and put that in the second spot etc. 
#this allows us to use the data we already have to create new samples in order to get a better error estimation. 
def bootstrap(W):
    W_boot = W.copy()
    for i in range(Ncf):
        a = np.random.randint(0,Ncf)
        W_boot[a] = W[i]
        
    return W_boot
#this overwrites a certain configurations values, meaninf we could get duplicates in W_sim.  

#function that calculates the product of link variables for one side of a arbitrary length Wilson loop.
#Inputs: U - Array of link variables. x,y,z,t - current location on the lattice where the side of the loop starts
#        sep - vector holding the separation distance of the corners of the Wilson loop
#Outputs: L - product of link variables representing the one side of the wilson loop
#This function now works for non-planar wilson loops, applying link variables in order of x,y,z,t to get to the point we want
#Generally, we will have one side of the loop being purely time separation, then the perpendicular side being purely space separation
#but if the separation of a certain dimension is 0, that dimension's loop does nothing to L. 

def LoopSide(U,x,y,z,t,sep,dex,N):
    L = np.eye(3,dtype=complex)
    loc = np.array([x,y,z,t],dtype=int)
    loc_m = np.zeros(4,dtype=int)
    loc_m = loc.copy()
    
    #for loop that loops through possible combinations of steps for any given separation vector
    #in this loop we then call shuffle and it shuffles the order once. shuffle then has to shuffle the order just once, and cyclicly so that we can call it as many times as we need, and it will eventually get back to the original order.
    for d in dex:
        dir_vec = np.zeros(4,dtype=int)
        dir_vec[d] = 1
        for i in range(sep[d]):
            L = L@U[loc_m[0],loc_m[1],loc_m[2],loc_m[3],d]
            loc_m = (loc_m + dir_vec)%N
            
    return L


#in the loops that we want, we go back the same way we came on the other side of the loop, so this is the same function as above,
#but we reindex U and made sure to dagger it to get the proper link variable product for our loop.
def LoopSide_Inv(U,x,y,z,t,sep,dex,N):
    L = np.eye(3,dtype=complex)
    loc = np.array([x,y,z,t],dtype=int)
    loc_m = np.zeros(4,dtype=int)
    loc_m = loc.copy()
    dex_r = [dex[-1],dex[-2],dex[-3],dex[-4]]
    
    for d in dex_r: #this loops backward over the dimensions so we get the proper order on the return trip along our full loop
        dir_vec = np.zeros(4,dtype=int)
        dir_vec[d] = 1
        for i in range(sep[d]):
            loc_m = (loc_m-dir_vec)%N
            L = L@dagger(U[loc_m[0],loc_m[1],loc_m[2],loc_m[3],d]) 
    
    return L
    
    
#function that calculates all possible Wilson loops for some given r and t dimensions. 
#Inputs: U, array of link variables. r, length of loop in any given spatial direction. t, dimension of loop in time direction
#outputs: a scalar, WL that is the value of the wilson loop, evaluated as the real part of the trace of the product of link variables
#for any given line in the time direction and length in the spatial direcitons (r) there are 6 different planar loops to be evaluated. 
#for each spatial dimension there are two, one going CW and one going CCW. so we are then creating a sum of these 6 possible loops for any given time and radius. 
#direction of the loops arent specified when given only a time and radius, so we must calculate both of them
def WL_rxt(U,sep_t,sep,N):
    WL = 0  #there are 8 total lines per spatial dimension, four for CW, 4 for CCW. 
    sep_T = np.array([0,0,0,sep_t]) #a purely temporal separation
    sep_r = np.array([sep[0],sep[1],sep[2],0]) #a purely spatial separation
    dex = [0,1,2,3]
    
    t_line_f = np.zeros((3,3),dtype=complex) #time line forward CCW
    r_line_f = np.zeros((3,3),dtype=complex) #spatial line forward CCW
    t_line_b = np.zeros((3,3),dtype=complex) #time line backward CCW
    r_line_b = np.zeros((3,3),dtype=complex) #spatial line backward |
    
    #Then for the CW loop, they are the same lines but backward, so the forward direction then corresponds to the Other Side loop above
    t_line_CW_f = np.zeros((3,3),dtype=complex) 
    r_line_CW_f = np.zeros((3,3),dtype=complex)
    t_line_CW_b = np.zeros((3,3),dtype=complex) 
    r_line_CW_b = np.zeros((3,3),dtype=complex)   
    
    sep_p = np.array(list(itertools.permutations(sep)))
        
    #for each spatial dimension we iterate through the lattice to calculate all possible rxt loops through all possible permutations of x y and z separation that give the same r. 
    for s in sep_p:
        sep_r = np.array([s[0],s[1],s[2],0])
            
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    for t in range(N):
                        loc = np.array([x,y,z,t],dtype=int)
                        loc_c = (loc + sep_r)%N #this defines the location of the end of the line  that starts at [x,y,z,t] in the spatial direction
                        t_c = (t+sep_t)%N #This defines the location that is furthest away in the time direction. 
                        
                        #CCW
                        t_line_f = LoopSide(U, x, y, z, t, sep_T,dex,N) #3 corresponds to the time direction. length of this line is the function input T.
                        r_line_f = LoopSide(U, x, y, z, t_c, sep_r,dex,N) #s is the direction of the line, the input r is how long the line is
                        t_line_b = LoopSide_Inv(U, loc_c[0], loc_c[1], loc_c[2], t_c, sep_T,dex,N) #the location inputs of the line functions is always the starting point
                        r_line_b = LoopSide_Inv(U, loc_c[0], loc_c[1], loc_c[2], t, sep_r,dex,N) 
                        
                        #CW | from the same starting point [x,y,z,t] we now go CW around the same loop we did for CCW
                        r_line_CW_f = LoopSide(U,x,y,z,t,sep_r,dex,N)
                        t_line_CW_f = LoopSide(U, loc_c[0], loc_c[1], loc_c[2], t, sep_T,dex,N)
                        r_line_CW_b = LoopSide_Inv(U, loc_c[0], loc_c[1], loc_c[2], t_c, sep_r,dex,N)
                        t_line_CW_b = LoopSide_Inv(U, x, y, z, t_c, sep_T,dex,N)
                        
                        #now with each line defined we construct the loops
                        WL += np.real(np.trace(t_line_f@r_line_f@t_line_b@r_line_b)) \
                            +np.real(np.trace(r_line_CW_f@t_line_CW_f@r_line_CW_b@t_line_CW_b))
                      
    # the value we want from this function is the expectation value of some type of Wilson loop. this means we need to average our 
    # values based on the number we calculated for this specific time and radius. Each point on the lattice has 3 spatial dimensions
    # and each spatial dimension has 2 loops calculated from it at each point. this means we calculated N**4 (number of lattice sites) times 6 (6 possible loops per lattice site)
    WL = WL/(6.*3.)/N**4/2
    #print(WL)
    return WL    

#Get smeared U's 

#Function that evaluates the discretized gauge-covariant derivative in a certain direction rho and sums all possible rhos to get full Delta^2 on U
#Inputs: U, array of all link variables. indicies of the location and direction of the link variable we want to take the derivative of. Delta_rho on U is dependent on the specific link variable, so Delta is as well
#outputs: Delta(2) on U. 
#we will iterate through all points on the lattice, and all directions for rho != mu to get the derivative of each link variable as a new array
#this will be used multiple times in a row in order to smear our U's 
def Gauge_Cov(U,N):
    Delta = np.zeros((N,N,N,N,4,3,3),dtype=complex)
    for x in range(N):
        for y in range(N):
            for z in range(N):
                for t in range(N):
                    for mu in range(4):
                        mu_vec = np.zeros(4,dtype=int)
                        mu_vec[mu] = 1
                        
                        loc = np.array([x,y,z,t],dtype=int)
                        loc_mu = (loc + mu_vec)%N
    
                        for rho in range(4):
                            rho_vec = np.zeros(4,dtype=int)
                            rho_vec[rho] = 1
                        
                            loc_rho = (loc + rho_vec)%N
                            loc_mrho = (loc - rho_vec)%N
                            loc_mrho_mu = (loc - rho_vec + mu_vec)%N
                        
                            Delta_rho = (U[x,y,z,t,rho]@U[loc_rho[0],loc_rho[1],loc_rho[2],loc_rho[3],mu]@dagger(U[loc_mu[0],loc_mu[1],loc_mu[2],loc_mu[3],rho]) \
                                         - 2*u0**2*U[x,y,z,t,mu] + dagger(U[loc_mrho[0],loc_mrho[1],loc_mrho[2],loc_mrho[3],rho])@U[loc_mrho[0],loc_mrho[1],loc_mrho[2],loc_mrho[3],mu]@U[loc_mrho_mu[0],loc_mrho_mu[1],loc_mrho_mu[2],loc_mrho_mu[3],rho])
                        
                            Delta[x,y,z,t,mu] += (Delta_rho/(u0**2*a**2))
    
    return Delta #this gives total delta for a single link variable U_mu(x)
              
#This function smears the spatial link variables n times, recursively. we apply the 'smear operator' (1 + eps*a**2*Delta) n times on U_mu(x). every time we 
#apply one power of this operator, we get a new array of link variables that has been smeared once. We then apply this operator on the once smeared link variables, and repeat this process
#n times to get the proper amount of times. This is recursive, so the base case is the first application of the smear differential operator, then we redefine what we get from that case 
#as the new U and apply the function process again. 
#Inputs: array of link variables, n number of smears we want to do
#outputs: a new array of smeared link variables. 
#we go through the entire lattice and smear each lnk variables n times. each smear produces a new array of link variables that 
def Smear(U,n,N):
    U_s = np.zeros((N,N,N,N,4,3),dtype=complex)
    Der = np.zeros((N,N,N,N,4,3),dtype=complex)
    if n == 1:
        U_s = U.copy()
        Der = Gauge_Cov(U,N)
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    for t in range(N):
                        for mu in range(3): # we want to smear only the spatial variables, so we only take mu from 1 to 3.
                            U_s[x,y,z,t,mu] = (U[x,y,z,t,mu].copy() + (eps*a**2)*Der[x,y,z,t,mu].copy())
                            
        return U_s
                            
    
    else:
        U = Smear(U, n-1,N)
        U_s = U.copy()
        Der = Gauge_Cov(U,N)
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    for t in range(N):
                        for mu in range(3):
                            U_s[x,y,z,t,mu] = (U[x,y,z,t,mu].copy() + (eps*a**2)*Der[x,y,z,t,mu].copy())
        
        return U_s


def Exact(x,r,y):
    return x[0]*r - x[1]/r + x[2] - y

def func_generator(r,a,b,c):
    return a*r - b/r + c

#Main 
#constants

Ncf = 10 #number of configurations
Ns = 100 #bootstrap shuffles
N_rx = 4
N_ry = 4
N_rz = 4
Nt = 4
a = 0.25
eps = 1/12
u0 = 0.797 

#this blurb will give us a list of the different mangitude separations we have. we will then all the separations we need to iterate through
#for the non planar loops being included, to make sure a certain separation distance is not duplicated. 
r_list = []
sep_list = []
for sep_x in range(N_rx):
    for sep_y in range(N_ry):
        for sep_z in range(N_rz):
            v = [sep_x,sep_y,sep_z]
            new_r = np.sqrt((sep_x)**2 + (sep_y)**2 + (sep_z)**2)
            if new_r == 0 or new_r in r_list:
                if sep_x == 3 and new_r == 3.0: #this is all I got for now to get (3,0,0) and (2,2,1) 
                    r_list.append(new_r)
                    sep_list.append([sep_x,sep_y,sep_z])
                else:
                    continue
            else:
                r_list.append(new_r)
                sep_list.append([sep_x,sep_y,sep_z])
        

radii = np.array(r_list,dtype=float) #we want it in an np array 
sep = np.array(sep_list,dtype = int)

W = np.zeros((Ncf,Nt,len(radii)))
W_sim = np.zeros((Ns,Ncf,Nt,len(radii)))
avg_W = np.zeros((Ns,Nt,len(radii)))
avg_W_sq = np.zeros((Ns,Nt,len(radii)))
W_err = np.zeros((Ns,Nt,len(radii)))

L = Lattice(6,True) #creating an N = 4 Lattice
L.Thermalize()
U6 = L.U


for alpha in range(Ncf): #this loop represents the different configurations we will have
    L.Corr()
        
    S = U6.copy()
    #S = Smear(U6,4,L.N) #as the exercise asks, we smear the spatial link variables 4 times
    S = L.Smear(U6,4)
    
    #this iterates through possible dimensions for our rxt Wilson Loops. 
    for t in range(1,Nt): #we do not want 0 to be an option for length so we start the loop at 1
        for r in range(len(radii)): #radii starts at 1 so we start the loop at 0. 
            W[alpha,t,r] = WL_rxt(S,t,sep[r],L.N)  #we assign W of this configuration to be based on the dimensions of the Wilson Loop
            
    print('Measured configuration ',alpha+1)
    print('%.4f Seconds'%(time.time() - start_time))

#using bootstrap to create Ns different configuration orders using the values we already have
#W_sim is a 4D array that holds every bootstrap shuffle, configuration, r and t. 
W_sim[0] = W
for i in range(Ns):
    W_sim[i] = bootstrap(W)
      
phi = np.zeros((Ns,len(radii)),dtype=float) #potential 
phi_err_2 = np.zeros((Ns,len(radii)),dtype=float)
phi_sq = np.zeros((Ncf,len(radii)),dtype=float)
phi_avg = np.zeros(len(radii),dtype=float) #potential averaged over bootsrap shuffles
phi_avg_sq = np.zeros(len(radii),dtype=float)
phi_err_1 = np.zeros(len(radii),dtype=float)
phi_err_3 = phi_err_1.copy()
phi_err = phi_err_1.copy()

#we have a set of W's for every configuration, and for every possible separation. we now want to average over the configurations
for beta in range(Ns):
    for t in range(1,Nt):
        for r in range(len(radii)):
            for alpha in range(Ncf): 
                avg_W[beta,t,r] += W_sim[beta,alpha,t,r]
                avg_W_sq[beta,t,r] += W_sim[beta,alpha,t,r]**2
            avg_W[beta,t,r] = avg_W[beta,t,r]/Ncf
            avg_W_sq[beta,t,r] = avg_W_sq[beta,t,r]/Ncf
            
            W_err[beta,t,r] = ((avg_W_sq[beta,t,r] - avg_W[beta,t,r]**2)/Ncf)**(1/2)
            
#calculation of potential for each separation for each bootstrap.
for beta in range(Ns):
    for r in range(len(radii)):
        phi[beta,r] = abs(avg_W[beta,Nt-2,r]/avg_W[beta,Nt-1,r])
        phi_err_2[beta,r] = phi[beta,r]*np.sqrt((W_err[beta,Nt-2,r]/avg_W[beta,Nt-2,r])**2 + (W_err[beta,Nt-1,r]/avg_W[beta,Nt-1,r])**2)

#this potential is averaged over each bootstrap, then the variance is computed.
for r in range(len(radii)):
    for beta in range(Ns):
        phi_avg[r] += phi[beta,r]
        phi_avg_sq[r] += phi[beta,r]**2
        phi_err_3[r] += phi_err_2[beta,r]
        
    phi_avg[r] = phi_avg[r]/Ns
    phi_avg_sq[r] = phi_avg_sq[r]/Ns
    phi_err_1[r] = ((phi_avg_sq[r] - phi_avg[r]**2))**(1/2)
    phi_err_3[r] = phi_err_3[r]/Ns
    
#now we have an error in phi from averaging over the configurations and calculating phi, we also have one from the variance in phi from the bootstrap configs
for r in range(len(radii)):
    phi_err[r] = phi_err_1[r] + phi_err_3[r]
  
x = np.ones(3)
fit_lsq = least_squares(Exact,x,loss='soft_l1',args=(radii,a*phi_avg))
r_gen = np.linspace(0.2,4,500)
V_lsq = func_generator(r_gen, *fit_lsq.x)
    

fig,ax = plt.subplots()
ax.errorbar(radii, phi_avg*a, yerr = phi_err_1*a,fmt = '.', label = 'MC',color='darkviolet')
ax.plot(r_gen,V_lsq,label='regression',linestyle='dashed',linewidth='1',color='lightskyblue')
ax.set_xlabel('r/a')
ax.set_ylabel('a*V(r)')
ax.set_xlim(0,4)
ax.set_ylim(0,2.5)
ax.set_title('Static quark potential as a function of quark separation')
ax.legend()


print('%.4f Seconds'%(time.time() - start_time))






























