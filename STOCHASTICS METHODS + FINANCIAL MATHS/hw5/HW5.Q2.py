#hw5.p3

from pylab import *
from scipy.special import binom
from scipy.stats import norm
import matplotlib.pyplot as plt

"""
GTM: You are supposed to plot Gaussian for the same x range as Binomial plots
     You are supposed to plot for p=0.2 and p=0.5... (-0.5)
given points: 4.5
"""

a=[]
b=[]

#definations
def binomial_distribution(n,p):
    for i in range(0,n+1):
        """gtm: where did you get this p value?"""
        #p=0.75
        q=1-p
        x1 = binom(n, i) * (p ** i) * ((1 - p) ** (n-i))
        x2 = sqrt(n*p*(1-p))*x1
        b.append(x2)

def xaxis_correspond(n,p):
        for i in range(0,n+1):
            #p=0.75
            k1 = (i-n*p)/(sqrt(n*p*(1-p)))
            a.append(k1)
            
#n=10  
p=0.2          
binomial_distribution(10,p)
xaxis_correspond(10,p)
plt.plot(a, b, 'go')
#n=100
binomial_distribution(100,p)
xaxis_correspond(100,p)
plt.plot(a, b, '.r')
#standard Gaussian
x=linspace(-10,10,100)
plt.plot(x,norm.pdf(x, 0, 1),'y')
title("green:10, red=n:100, yellow=Standard Gaussian")
plt.show()

#comment:  AS VALUE OF n INCREASES, THE GRAPH TENDS TO LOOK LIKE STANDARD GAUSSIAN
