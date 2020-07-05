#HW6.Q1

from pylab import * 
from scipy.stats import norm
import numpy as np
import random
import math  
from random import randint
import matplotlib.pyplot as plt

"""
GTM: Wrong binomial tree computation and wrong plots (-1)
     Wrong comment due to wrong formulation -0.5
given points: 8.5
"""

T=1
#base=10
base=1
part = 1/500
mu1 = 0.2 
mu3 = 0.2
sigma1 = 0.6 
sigma3 = 0.6
mu2 = 0.6 
mu4 = 0.6
sigma2 = 0.2 
sigma4 = 0.2
n = 500
a=[]
b=[]
for i in range (0,1000):
    t = np.linspace(0, T, 500)
    K1 = np.random.standard_normal(size = n) 
    K1 = 0+np.cumsum(K1)*np.sqrt(part) 
    K2 = base*np.exp((mu1-0.5*sigma1**2)*t + sigma1*K1)
    K3 = base*np.exp((mu2-0.5*sigma2**2)*t + sigma2*K1)
    a.append(K2)
    b.append(K3)

    
K6=np.mean(a, axis=0)
K7=np.std(a, axis=0)
K8=np.mean(b, axis=0)
K9=np.std(b, axis=0)

plt.figure()
xlabel("time")
title("mu=0.2, sigma=0.6, geometric-brownian")
plot(K6)
plot([x + y for x, y in zip(K6, K7)], 'r')
plot([x - y for x, y in zip(K6, K7)], 'r')
for i in range(0, 10):
    plot(a[randint(0,999)])

plt.figure()
plot(K8)
plot([x + y for x, y in zip(K8, K9)], 'r')
plot([x - y for x, y in zip(K8, K9)], 'r')
xlabel("time")
title("mu=0.6, sigma=0.2, geometric-brownian")
for i in range(0, 10):
    plot(b[randint(0,999)])

#...............................................................................

""" GTM: wrong computation, one way to do that 
     i) random_sample in interval [0,1], true = 1, false = 0
     ii) creating matrix of u with probability p, d with probability 1-p
           dS = d + (u-d)*(random_sample((M,N))<p) 
     iii)create the Stock price paths
            S = c_[ones(M), cumprod(dS, axis=1)]
"""

x=linspace(0,1,500)
c=[]
d=[]
K10=np.exp(mu3*T/n)
K11=np.exp(sigma3*((T/n)**0.5))
K12=1/K11
K13=np.exp(mu4*T/n)
K14=np.exp(sigma4*((T/n)**0.5))
K15=1/K14
K=(K10-K12)/(K11-K12)
e=[]
f=[]
    
for i in range(0, 1000):
    K17=base
    K18=base
    c=[]
    d=[]
    c.append(K17)
    d.append(K18)
    for j in range (0,500):
        z=np.random.binomial(1,K,1)
        if (z==1):
            K17=K17*K11
            K18=K18*K14
            c.append(K17)
            d.append(K18)
        else:
            K17=K17*K12
            K18=K18*K15
            c.append(K17)
            d.append(K18)
    e.append(c)
    f.append(d)

K19=np.mean(e, axis=0)
K20=np.std(e, axis=0)

plt.figure()
plot(K19, 'b')
plot([x + y for x, y in zip(K19, K20)], 'r')
plot([x - y for x, y in zip(K19, K20)], 'r')
xlabel("time")
title("mu=0.2, sigma=0.6, bionomial-tree")

for i in range(0, 10):
    plot(e[randint(0,999)])
    
K21=np.mean(f, axis=0)
K22=np.std(f, axis=0)

plt.figure()
plot(K21, 'b')
plot([x + y for x, y in zip(K21, K22)], 'r')
plot([x - y for x, y in zip(K21, K22)], 'r')
xlabel("time")
title("mu=0.6, sigma=0.2, bionomial-tree")


for i in range(0, 10):
    plot(f[randint(0,999)])
#...............................................................................    
plt.figure()
plot(K6)
plot([x + y for x, y in zip(K6, K7)], 'r')
plot([x - y for x, y in zip(K6, K7)], 'r')
plot(K19, 'b')
plot([x + y for x, y in zip(K19, K20)], 'r')
plot([x - y for x, y in zip(K19, K20)], 'r')
xlabel("time")
title("mu=0.2, sigma=0.6")
for i in range(0, 10):
    plot(a[randint(0,999)])
for i in range(0, 10):
    plot(e[randint(0,999)])
    
plt.figure()
plot(K8)
plot([x + y for x, y in zip(K8, K9)], 'r')
plot([x - y for x, y in zip(K8, K9)], 'r')
plot(K21, 'b')
plot([x + y for x, y in zip(K21, K22)], 'r')
plot([x - y for x, y in zip(K21, K22)], 'r')
xlabel("time")
title("mu=0.2, sigma=0.6")
for i in range(0, 10):
    plot(b[randint(0,999)])
for i in range(0, 10):
    plot(f[randint(0,999)])
    
show()

"when mu=0.2, sigma =0.6, mean,mean+sd and mean-sd curves tend to diverge, while when mu =0.6,sigma=0.2, they tends to overlap "
