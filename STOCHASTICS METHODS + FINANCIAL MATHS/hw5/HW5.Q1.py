#hw5.p1

from pylab import *
from scipy.stats import linregress
from scipy.special import factorial

"""
GTM: In loglog plot, we do not compute the logarithmic values of
     the error. Loglog plot sketch the graph in logarithmic
     scale so that order of scale can be detected easily. When one looks
     at the axes, it is visible that actual error and period values are
     replaced on them in log-scaling.
     wrong loglog plotting ... (-0.5)
given points: 4.5
"""

#striling approximation numerically
n = linspace(1, 50, 50)
def stirling_aprroximation_fxn(n):
    fxn = sqrt(2*pi*n)*((n/e)**n)
    return fxn

#for relative error
a=[]
for i in range(0,50):
    x1= (factorial(n[i], exact=False)-stirling_aprroximation_fxn(n[i]))/stirling_aprroximation_fxn(n[i])
    #x2=log(x1)
    #a.append(x2)
    a.append(x1)
    
#plottings    
#plot(log(n),a,'r')
loglog(n, a, '*')
title('logarithmic plot of the relative error')
ylabel("error")
xlabel("log(n)")
print(linregress(a, log(n)))
print('slope=-0.996; Yes,straight line is obtained; C;next order is 1/12')
show()
