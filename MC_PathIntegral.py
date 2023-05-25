# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 13:36:31 2023

@author: Hilde
"""

import numpy as np
import matplotlib.pyplot as plt
import vegas as vegas
import time

start_time = time.time()

a = 0.5
N = 8
m = 1
A = (m/(2*np.pi*a))**(N/2)
samp = 100000

#Function, coming from the functools packet,that given a function gives back the function
#computed in one of the variables of the function
def partial(func, *args, **keywords):
    def newfunc(*fargs, **fkeywords):
        newkeywords = keywords.copy()
        newkeywords.update(fkeywords)
        return func(*args, *fargs, **newkeywords)
    newfunc.func = func
    newfunc.args = args
    newfunc.keywords = keywords
    return newfunc


def exact(x):
    E0 = 0.5
    T = 4
    return np.exp(-E0*T)*(np.exp(-(x**2)/2)*(np.pi**(-1/4)))**2

#classical action, input is a 'function' y, which represents a possible path, S is evaluated at all discrete points on the path
#we will integrate over each path later, passing a new path through this function each integral. 
#x is the endpoints
def S(x,y):
    s = (m/(2*a))*(y[0] - x)**2 + a*((x)**2)/2
    for j in range(N-1):
        if j == N-2:
            s+= (m/(2*a))*(x - y[j])**2 + a*((y[j]**2)/2)
        else:
            s+= (m/(2*a))*(y[j+1] - y[j])**2 + a*((y[j]**2)/2)
        
    return A*np.exp(-s)

y = np.linspace(0,2,10)
result = np.zeros(10)
err = np.zeros(10)
z = np.linspace(0,2,100)
ex = np.zeros(100)

for k in range(10):
    integ = vegas.Integrator(7*[[-5,5]])
    integration = integ(partial(S,y[k]), nitn=10,neval=100000)
    result[k] = integration.mean
    err[k] = integration.sdev
    #print(y[k],result[k],err[k])

for k in range(100):
    ex[k] = exact(z[k])
  
fig,ax = plt.subplots()
ax.errorbar(y,result,yerr=err,fmt='o',label='path integral')
ax.plot(z,ex,label='exact')
ax.set_xlabel('x')
ax.set_ylabel(r'$\langle x|e^{HT}|x \rangle$')
ax.set_title('Harmonic oscillator')
ax.legend()

print('%.4f Seconds'%(time.time() - start_time))

