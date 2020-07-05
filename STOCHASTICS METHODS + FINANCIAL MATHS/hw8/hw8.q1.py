#Q.1.

from pylab import * 
from scipy.stats import norm
import numpy as np
import random
import math  
import matplotlib.pyplot as plt

"""
GTM: part a: is not derived... -4
             If your formulation were right in the code,
             I would count part a as 4 points as well.
     part b: wrong formula... -2
             missing 0 entry in W... -0.5 
             
given points: 1.5
"""

seed(1)
µ=0.5
σ=1.8
#n=1000
n=20000
dt = 1/n
t1=np.linspace(0,1,n)
dW = np.random.normal(0, 1, n)*np.sqrt(dt)
"""GTM: why not adding 0 entry?"""
#W = np.cumsum(dW)
W = np.r_[0, np.cumsum(dW)][:-1]
K1 = (µ-0.5*σ**2)*t1 + σ*W 
S0=1
S1= S0*np.exp(K1)
K2= ((1+t1)**2)*np.sin(S1)
plot(t1,K2, 'r')

 
"""
GTM: this formula is wrong...
In the homework, the function was updated and you did not 
update the Euler Maruyama formula according to that.""" 
a1=np.zeros(n)
a1[0]=S0
for i in range (0, n-1):
    a1[i+1]=a1[i]+(1/(1+t1[i]) + µ/2 - σ*σ/8)*a1[i]*dt + σ/2*a1[i]*dW[i]

plot(t1,a1, 'g')
xlabel('time interval ')
ylabel('approximaion')
plt.title("Ito Formula")
legend(('F(X,t)','Approximation: DE solution '), loc ='best')
show()
