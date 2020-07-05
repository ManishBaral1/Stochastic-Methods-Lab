#hw8.q3

from pylab import * 
from scipy.stats import norm
import numpy as np
import random
import math  
import matplotlib.pyplot as plt

"""
GTM: Dont you run the code before sending?
     You just coppied the first part from problem 1 and 
     in the code, it is not clear to you what the main function
     is... -3.5
given points: 0.5
"""

#Geometric Brownian Motion
µ=2
σ=1
m0=1
T=1
n=256
dt = T/n
t1=np.linspace(0,T,n)
dW = np.random.normal(0, 1, n)*np.sqrt(dt)
"""GTM: why not adding 0 entry?"""
W = np.cumsum(dW)
K = (µ-0.5*σ**2)*t1 + σ*W 
brownian_motion= m0*np.exp(K)
plt.plot(t1,brownian_motion, 'r')

"""
GTM: create a function which produces np.sum(np.square(dW)) 
    wrt to different number of steps, then for a range of
    steps, compute the vectorized function.
"""

r=2
def conv(n,r): #convergence
    a=[]
    for i in range(1, n-1):
        a.append((brownian_motion(n,r))**2)
    return np.a
    
""" GTM: brownian_motion is not a function, your function is conv"""
print("the constant is = ", brownian_motion(256,2))

#note: replacement of value of integer =n=256 with inf (infinity) is needed to get exact constant
