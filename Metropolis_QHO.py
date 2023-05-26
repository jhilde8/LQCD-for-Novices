# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 15:11:43 2023

@author: Hilde
"""
import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()

a = 0.5
N = 20
m = 1
A = (m/(2*np.pi*a))**(N/2)
T = 4
eps = 1.4
Ncor = 20
Ncf = 5000


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

#calculates action at one lattice site, since the action only depends on the lattice sites next to it, we can get dS
#without having to calculate the action on the entire lattice each time
def S(j,x):
    jp = (j+1)%N #next lattice site, %N enforces periodic BC
    jm = (j-1)%N 
    return a*x[j]**2/2 + x[j]*(x[j]-x[jp]-x[jm])/a

#function that calculates G, using EQ 36
def compute_G(x,n):
    g = 0
    for j in range(0,N):
        g += x[j]*x[(j+n)%N]
        
    return g/N
            
def compute_G3(x,n):
    g = 0
    for j in range(0,N):
        g += (x[j]**3)*(x[(j+n)%N]**3)
    
    return g/N

#function that follows the method explained on page 9
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
    #so we have N values of G for each path x_p (Ncf of them), we then want to average these G's over each path 
    for n in range(0,N): #for each n value there are Ncf values of G for each path x_p. 
        avg_G = 0 
        avg_G_sqr = 0
        for p in range(0,Ncf): #we sum all values of G for this specific n, basically summing down the columns of the G matrix
            avg_G += G[p,n]
            avg_G_sqr += G[p,n]**2
            
        Gn[n] = avg_G/Ncf #quantity we3 are calculating
        G_squared[n] = avg_G_sqr/Ncf #used for error 
        #print(avg_G,n)
        
    return Gn,G_squared


G = MC_avg(compute_G) #function returns a tuple since we want G squared average and G average
Gn1 = G[0] #G average
Gsq1 = G[1] #G sqaured average
E_real = np.ones(N) #actual solution for the HArmonic oscillator
t = np.linspace(0,20,N) #Time 

G3 = MC_avg(compute_G3)
Gn3 = G3[0]
Gsq3 = G3[1]


def QOI(Gn,Gsq):
    ErrG = np.zeros(N) #Setting up an array for the error in G and the error in E
    ErrE = np.zeros(N)
    dE = np.zeros(N) #change in energy 
    for n in range(0,N-1):
        ErrG[n] = ((Gsq[n] - Gn[n]**2)/Ncf)**(1/2) #Error in G, sigma squared, EQ 34
        dE[n] = np.log(abs(Gn[n]/Gn[n+1]))/a #change in energy, EQ 38 
        ErrE[n]=(((ErrG[n]*Gn[n+1])/(a*Gn[n]))**2+((ErrG[n+1]*Gn[n])/(a*Gn[n+1]))**2)**(1/2)
        
    return dE,ErrE

E1 = QOI(Gn1,Gsq1)
dE1 = E1[0]
ErrE1 = E1[1]

E3 = QOI(Gn3,Gsq3)
dE3 = E3[0]
ErrE3 = E3[1]
    
fig,ax = plt.subplots()
ax.errorbar(t*a,dE1,yerr=ErrE1,fmt = '.',label='Numerical')
ax.plot(t,E_real,label='Actual')
ax.set_xlim(0,3.5)
ax.set_ylim(0,2.5)
ax.set_xlabel('t')
ax.set_ylabel(r'$\Delta E$')
ax.set_title('Metropolis algorithm for QHO - x')
ax.legend()

fig2,ax2 = plt.subplots()
ax2.errorbar(t*a,dE3,yerr=ErrE3,fmt = '.',label='Numerical')
ax2.plot(t*a,E_real,label='Actual')
ax2.set_xlim(0,3.5)
ax2.set_ylim(0,2.5)
ax2.set_xlabel('t')
ax2.set_ylabel(r'$\Delta E$')
ax2.set_title('Metropolis algorithm for QHO - x^3')
ax2.legend()

print('%.4f Seconds'%(time.time() - start_time))


