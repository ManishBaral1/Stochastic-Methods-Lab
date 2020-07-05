#hw5.p3

from pylab import *
from scipy.special import binom
from scipy.stats import norm
import statistics as st

"""
GTM: Not providing proper comment... (-0.5)
     It is obvious that the plot is linear but
     try to comment in the light of central limit theorem.
given points: 4.5
"""

#Binomial Distribution
Sample1 = np.random.binomial(10000, 0.5, 10000)
""" GTM: wrong computation..."""
k1= (Sample1-np.mean(Sample1))/np.std(Sample1)
print(k1)

#Standard Normal Distribution
Sample2 = np.random.normal(0, 1, 10000)
print(Sample2)

Sample1=sorted(k1)
Sample2=sorted(Sample2)

plot(Sample2,Sample1,'ro')
ylabel("Binomial Distribution")
xlabel("Standard Normal Distribution")
title("Q-QPlot(Compare 2 probability distribution)")
show()


#COMMENTS: PLOT FOLLOWS STRAIGHT LINE, ALTHOUGH SOME POINTS OUTSIDE

