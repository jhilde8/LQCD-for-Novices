# -*- coding: utf-8 -*-
"""
Created on Fri May 26 16:52:34 2023

@author: Hilde
"""

import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()


a = 0.5
omega = 1
N = 20 #points on the time lattice
Nbin = 20
m = 1
A = (m/(2*np.pi*a))**(N/2)
T = 4
eps= 1.4
Ncor = 20
Ncf = 10000 #number of paths
Ns =100

#Method of estimating statistical error. We take the G matric found in the MC process and randomly generate a new G matrix using the same numbers 
#as the old one but in a different order. This allows a quick way to simulate many different G matricies for different paths
# we then average over the different simulations, and use that as our G to calculate energy
def bootstrap(G):
    Nb = len(G)
    G_boot = np.zeros([Nb,N])
    for i in range(Nb):
        alpha = int(np.random.uniform(0,Nb))
        G_boot[i] = G[alpha]
        
    return G_boot

def binning(G,binsize):
    G_binned = []
    for i in range(0,len(G),binsize):
        G_avg = 0
        for j in range(0,binsize):
            G_avg += G[i+j]
        G_binned.append(G_avg/binsize)
    
    return G_binned

def Err(Gn,Gsq):
    sig = np.zeros(N)
    for n in range(N):
        sig[n] = ((Gsq[n] - Gn[n]**2)/Ncf)**(1/2)
    return sig

#inputs current path/configuration, outputs new path/config (or the old one) based on the change in the action
#sweeps through the configuration, at each site it proposes a change of eps for that part of the path, calculates action
#if action decreases or is greater than some probability distribution, we keep the change, if not, we discard the change. 
def update(x):
    for j in range(0,N): #we iterate through each lattice site and change the path by e at each element, calculating the action each time
        xp = x[j] 
        Sp = S(j,x)
        x[j] = x[j] + np.random.uniform(-eps,eps)
        dS = S(j,x) - Sp
        if (dS > 0) and (np.exp(-dS) < np.random.uniform(0,1)):
            x[j] = xp
    return x

def Delta2(x,j):
    jp = (j+1)%N
    jm = (j-1)%N
    return (x[jp] - 2*x[j] + x[jm])/(a**2)


#calculates action at one lattice site, since the action only depends on the lattice sites next to it, we can get dS
#without having to calculate the action on the entire lattice each time
def S(j,x):
    jp = (j+1)%N #next lattice site, %N enforces periodic BC
    jm = (j-1)%N 
    jpp = (j+2)%N
    jmm = (j-2)%N
    #xddot = np.zeros(N)
    #for i in range(0,N):
        #xddot[i] = Delta2(x,i)
    
    #return a*(a*x[j]**2/2-(1/2)*x[j]*(Delta2(x,j)-((a**2)/12)*Delta2(xddot,j)))

    return (a**2/2)*x[j]**2*(1+((a)**2/12)) + (1/(24*a))*(m*x[j])*((-16)*(x[jp] + x[jm])+(x[jpp] + x[jmm])+(15)*x[j])


#function that calculates G, using EQ 36
def compute_G(x,n):
    g = 0
    for j in range(0,N):
        g += x[j]*x[(j+n)%N]
        
    return g/N
            

#function that follows the method explained on page 9
#we return the calculated value, G, as a 2D array, with rows representing paths, columns representing time steps or lattice sites.
#So G[3,5] would be the value of G on path 4 at lattice site 6.
def MC_avg(cG):
    Gn = np.zeros(N)
    G = np.zeros([Ncf,N])
    G_squared = np.zeros(N)
    x = np.zeros(N) #initial path is all 0's
    for j in range(0,5*Ncor): #Thermalize x
        update(x)
    for p in range(0,Ncf): #repeat process Ncf times
        for j in range(0,Ncor): #update path Ncor times, this imposes statistical independence between paths
            update(x)
        for n in range(0,N): #calculate G for all values of n with the current path x
            G[p,n] = cG(x,n) #put each g in a 2D array with each path as the rows and G for each n as the columns  
    
    #GBin = binning(G,Nbin) #bins the paths by averaging over a small amount of them to reduce computational cost later
    
    return G

G_sim = np.zeros([Ns,Ncf,N]) #3D array, first index represents a version of G's shuffled via the bootstrap function, next two represent alpha, the path, and n, the number of time steps
G_sim[0] = MC_avg(compute_G) #storing the first G matrix, as its the one we calculated using the MC process
avgG = np.zeros([N,Ns])

#now we bootstrap to fill the rest of this matrix
for i in range(Ns):
    G_sim[i] = bootstrap(G_sim[0])

#we now average G over each bootstrap shuffle, this gives us a new G that is a 2D array, set up with configs as rows and time steps/lattice sites as columns
for k in range(Ns): 
    for n in range(0,N):
        avgG[n][k] = 0 # 2D array to hold the average over paths of G, we average over paths for each bootstrap configuration, so time step is the columns, config number is the rows
        for p in range(0,Ncf):
            avgG[n][k] += G_sim[k][p][n] #we add G at each time step for each path, effectively adding down the columns to get average G per time step
            
        avgG[n][k] = avgG[n][k]/Ncf #then for the average we divide by the number of paths, the number of paths was binned into bins of Nbin, so we divide by Ncf//Nbin
        

#initializing important energy quantities
avgdE = np.zeros(N-1)
avgdE_sq = np.zeros(N-1)
dE = np.zeros([N-1,Ns])
ErrdE = np.zeros(N-1)
        
#we calculate change in energy from the bootstrapped average of G
#we calculate dE for each configuration, so we loop over n and calculate dE for each lattice site, then loop over Ns to do that for each configuration
for k in range(0,Ns):
    for n in range(0,N-1):
        dE[n][k] = np.log(abs(avgG[n][k]/avgG[n+1][k]))/a

#then we average dE for each configuration to get a better estimate of dE.
for n in range(0,N-1):
    avgdE[n] = 0
    avgdE_sq[n] = 0
    for k in range(Ns):
        avgdE[n] += dE[n][k]
        avgdE_sq[n] += (dE[n][k])**2 
    avgdE[n] = avgdE[n]/Ns
    avgdE_sq[n] = avgdE_sq[n]/Ns
    ErrdE[n] = np.sqrt((avgdE_sq[n] - (avgdE[n])**2))

      
E_real = np.ones(N-1) #actual solution for the Harmonic oscillator
t = np.arange(0,9.5,a) #Time 


fig,ax = plt.subplots()
ax.errorbar(t,avgdE,yerr=ErrdE,fmt = '.',label='Numerical',color='darkviolet')
ax.plot(t,E_real,label='Actual',color='lightskyblue')
ax.set_xlim(0,3.5)
ax.set_ylim(0,2.5)
ax.set_xlabel('t')
ax.set_ylabel(r'$\Delta E$')
ax.set_title('Metropolis for QHO - ghost improved action')
ax.legend()

print('%.4f Seconds'%(time.time() - start_time))