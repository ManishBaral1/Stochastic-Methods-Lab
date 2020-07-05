#Q.NO.1:

from pylab import * 
from scipy.stats import norm
import numpy as np
import random
import math  
import matplotlib.pyplot as plt

"""  
GTM: The code is full of contradictions.
     
     part a: Due to below comments... -1
     part b and c: wrong understanding and formulations... -2
                   wrong convergence rate: -2
     If convergence rate is asked then you need to plot.
     
given points: 7
"""

#Geometric Brownian Motion
µ=2
σ=1
m0=1
T=1
n=2**8
dt = T/n
t1=np.linspace(0,T,n)
dW = np.random.normal(0, 1, n)*np.sqrt(dt)
""" GTM: why not adding 0 as first elemen of W?"""
#W = np.cumsum(dW)
W = r_[0, cumsum(dW[:-1])]
K = (µ-0.5*σ**2)*t1 + σ*W 
m1= m0*np.exp(K)
plt.plot(t1,m1, 'r', label='GBM')

def Euler_Mryma_mthd(n,r): #Euler Maruyama Method
    """GTM: First think 
    of the purpose: The following code is needed to create a new W wrt 
    different Dt, so that one can plot the strong and weak error as a function
    of Dt. Indeed to achieve this r should not be fixed: The function must be 
    a function of let say k and r=2^k.
    But this is for part b. In part a, you need to use the same W as in
    the geometric Brownian motion, and you already built W at the beginning."""
    r=2
    Dt=r*dt
    length=n/r
    m2=[m0]
    for i in range(0,int(length)):
        k= m2[i]+µ*m2[i]*Dt+σ*m2[i]*(np.sum(dW[(r*(i-1)+r):(r*i+r)]))
        m2.append(k)
    return m2[:-1]
r=2
t2=np.linspace(0,T,int(n/r))
plot(t2, Euler_Mryma_mthd(2**8, 2), 'g', label='EM')
plt.xlabel("t")
"""GTM: you have to use legend for this, not title"""
plt.title("a: red:Geometric Brownian Motion, green:Euler Maruyama Method")
plt.legend()
plt.show()

"""GTM: Rad the comment above"""
def Strng_odr_conv(n,r): #Strong order of convergence
    a=[]
    for i in range(1, n):
        """GTM: first take mean then append."""
        a.append(abs(Euler_Mryma_mthd(n,r)-m1[-1]))
    return np.mean(a)
print("b: strong_error_of_convergence(p) = ", log(Strng_odr_conv(2**8,2)), "divided by log (C*dt)")


def Wk_odr_conv(n,r): #Weak order of convergence
    b=[]
    for i in range(1, n,r):
        b.append(abs(mean(Euler_Mryma_mthd(n,r))-mean(m1[-1])))
    """GTM: why taking mean again"""    
    return np.mean(b)
print("c: weak_error_of_convergence(q) = ", log(Wk_odr_conv(2**8,2)), "divided by log (C*dt)")


#comments
''''
#b
If the covergent order is p and make each step t times smaller then the approximation error will decrease by t**p. 

#c
Compute the slope of the error in the logarithm axis using the formula log(y(µ)) = log(C) + q*log(µ)
and get the value of q. Approximate error is increasing.

'''

