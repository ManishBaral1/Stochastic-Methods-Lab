
from pylab import *
import scipy as sc
from scipy import optimize

"""
GTM: given point: 2
"""

m=2
n=10
p=75
f=100
c=0.10

def yield_to_maturity(y,m,n,p,f,c):
    k=0
    r=n*2 + 1
    for i in range (1, r):
        k=k+ ((c/m)/((1+y/m)**i))
    k=k + 1/((1+y/m)**(n*m))
    k=f*k
    k=k-p
    return k

print((sc.optimize.brentq(yield_to_maturity, -1,1, args=(m,n,p,f,c)))*100)

