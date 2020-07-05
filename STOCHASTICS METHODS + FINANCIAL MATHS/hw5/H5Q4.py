#hw5.p4
from pylab import * 
from scipy.stats import norm
import random
import math  

"""
GTM: 
given points: 5
"""

"""
GTM : HW5: 18.5 points in total
"""

#Ensemble of BMs:M BM paths
DW=normal(0,1,(1000,500))*1/sqrt(500) 
W= c_[zeros(1000), cumsum(DW, axis=1)]

#mean and standard deviation values
mn=np.mean(W, axis=0)
sd=np.std(W, axis=0)

r= random.sample(range(0, 999), 10)
for j in range (0, 10): #10 sample paths
    plot(W[r[j]])
    
#plottings    
plot(mn, 'g')
plot([x + y for x, y in zip(mn, sd)], 'r')
plot([x - y for x, y in zip(mn, sd)], 'r')
xlabel("time")
title("greenline= mean, redcurves= mean+sd, mean-sd")
show()
