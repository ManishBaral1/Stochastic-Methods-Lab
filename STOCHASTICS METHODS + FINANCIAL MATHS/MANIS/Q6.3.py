#HW6.Q3

from pylab import *
import numpy as np


""" 
GTM: Well done!
given points: 5
"""

seed(1)
#N = 300000 #N choosen as large as 300000
N = 10000
d_theta= normal(0, sqrt(1/N), size = N) 
Brownianmotion = r_[0, cumsum(d_theta)]
da = (d_theta[:-1:2] + d_theta[1::2])
m = np.arange(N/2)
Itointegral = cumsum(da*Brownianmotion[:-1:2])
Stratonovichintegral = cumsum(da*Brownianmotion[1::2])

plt.plot(m , Itointegral, 'b', label = "Ito")
plt.plot(m, Stratonovichintegral, 'r', label = "Stratonovich")
plt.plot(m, -Stratonovichintegral + Itointegral, 'y', label = "Difference")
plt.plot(m, Brownianmotion[:-1:2], 'g', label = "Brownian")
plt.legend()
plt.xlabel("time")
plt.title("Realization")
show()

"for large N, Ito and Stratonovich integrals tends to diverge"
