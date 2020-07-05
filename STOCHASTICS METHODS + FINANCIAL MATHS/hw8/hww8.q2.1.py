#hw8.q2.1 

from pylab import *

import numpy as np
import numpy.ma as ma
from scipy.stats import norm
import pandas as pd

""" 
GTM: You were supposed to find the volatility by using Black-Scholes
     formula on your own. If questions are not clear to you please ask.
given points: 1
"""

#American call option of Facebook Inc as for Nov 15, 2019 : https://finance.yahoo.com/quote/FB/options?p=FB&straddle=false

fb_data= pd.read_csv("datafb1.csv", sep = "/") #extracted data from above website to csv file

xlabel('strike_price')
ylabel('implied_volatility')
title("Facebook Inc Option: Implied Volatility vs Strike Price")

k1 = np.asarray(fb_data[['Strike']])
k2 = np.asarray(fb_data[['Implied Volatility']])

plot(k1,k2, '--*g')
plt.show()
