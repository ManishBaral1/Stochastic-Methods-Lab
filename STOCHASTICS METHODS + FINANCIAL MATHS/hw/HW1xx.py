
from pylab import *
import timeit
import numpy as np
from scipy.optimize import brentq
from scipy import *
import scipy as sc
from scipy import misc
N=300
C=120*arange(10, N+10)
P=15000

"""
GTM: given point: 6
"""

def fn(r):
    k=0
    for i in range (1, N+1):
        k=k+C[i-1]/((1+r)**(i))
    k=k-P
    return k
np.seterr(divide='ignore', invalid='ignore')       



def bisection_method(a,b,e):
    length=b-a
    while (length>e):
        length=b-a
        c=(b+a)*0.5
        if (fn(c)==0):
            return c
        elif (fn(a)*fn(c)<0):
            b=c
        else:
            a=c
    return c
a=0.01
b=1
e=0.0001
print( bisection_method(a,b,e))
print( timeit.Timer().timeit(number=1000), 's')


def newton_method(X0,X1): 
    while (abs(fn(X1))-10**(-6) > 0):
        
        X1=X0-(fn(X0)/(sc.misc.derivative(fn, X0,  dx=1e-6)))
        X0=X1
    return X1
X0=0.01
X1=1
print( newton_method(X0,X1))
print(timeit.Timer().timeit(number=1000), 's')

def secant_method(X0,X1):
    
    while (abs(fn(X1))-10**(-6) > 0):
        X=X1-(fn(X1)*(X1-X0))/(fn(X1)-fn(X0))
        X0=X1
        X1=X
    return X1
X0=0.1
X1=0.05
print(secant_method(X0,X1))
print( timeit.Timer( ).timeit(number=1000), 's')


def pythonbrentq_method(x): 
    return(sc.optimize.brentq(fn, -x,x))
x=1
print(pythonbrentq_method(x))    
print(timeit.Timer().timeit(number=1000), 's')


