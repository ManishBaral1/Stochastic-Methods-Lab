#FINAL_PROJECT

#QUESTION(a) + QUESTION(b)



import pylab as pl
import csv
import numpy as np
from math import log, sqrt
from scipy.special import comb
from pylab import *
from scipy.stats import norm
import pandas
import numpy 
from scipy.stats import norm
import matplotlib.pyplot as plt


def f1(j):
    i=np.arange(2,N,2**j)
    ddt=1/(len(i))
    def logreturn(i):
        return log(S1[i])-log(S1[i-2**j]) #log returns
    logreturnf=np.vectorize(logreturn)
    logret=(logreturnf(i))
    sample_logreturn=sum(logret)/(len(i))
    sample_sigma=(1/(len(i)-1))*sum((logret-sample_logreturn)**2)
    sigmah=sqrt(sample_sigma/ddt)
    muh= sample_logreturn/ddt + 0.5* (sigmah)**2
    return[muh,sigmah]

def f2(j):
    i=np.arange(2,N,2**j)
    ddt=1/(len(i)+1)
    def logreturn(i):
        return log(S1[i])-log(S1[i-2**j])#log returns
    logreturnf=np.vectorize(logreturn)
    logret=(logreturnf(i))
    sample_logreturn=sum(logret)/(len(i)+1)
    sample_sigma=(1/(len(i)+2))*sum((logret-sample_logreturn)**2)
    def norm(x):
        return (x-sample_logreturn)/sqrt(sample_sigma)#normalise the log returns
    normf=np.vectorize(norm)
    return np.sort(normf(logret))

def f3(j):
    i=np.arange(2,N,2**j)
    def logreturn(i):
        return log(S1[i])-log(S1[i-2**j])
    logreturnf=np.vectorize(logreturn)
    logret=(logreturnf(i))
    return (logret)


with open('MICRO.csv') as csvdata:#gets the stoke price microsoft from dec20, 2016 to dec 20, 2019
    read=csv.reader(csvdata,delimiter=',')
    a1=[]
    for r in read:
        S=r[1]
        a1.append(float(S))
S1 = a1
N=len(a1)
Normal=pl.normal(0,1,size=len(f2(1)))#normal estimates 
fig,ax=pl.subplots(1,3)
ax[0].plot(a1)
ax[1].plot(pl.sort(Normal),f2(1)) #QQ plot
ax[1].set_xlabel('normal distribution') 
ax[1].set_ylabel('normalized logret')
ax[1].legend(('QQ plot'),loc='upper right')
ax[2].acorr(f3(1),maxlags=60)  #AUTOCORRELATION PLOT
ax[2].set_xlabel('autocorrelation plot')
pl.tight_layout()
pl.show()
S1=a1
N=len(S1)
print("[mu,sigma]",f1(1))

#QUESTION NO (c)

r=1.5/100#US treasury(1 year) 
sigma=0.33953053460635674 #from question (a)
S=157#MICROSOFT price at 20 dec 2019
K=100#strike price from the option quotes 
t=1#1 year
mu=0.9604131585550888 #from question (a)
n=1000


def Black_Scholes(sigma,S,K,r,T):#balck scholes formula takes the call price 
    x= (log(S/K)+(r+(sigma**(2))*0.5))*(T/(sigma*sqrt(T)))
    X1= norm.cdf(x,loc=0, scale=1)
    X2= norm.cdf(x-(sigma*sqrt(T)),loc=0,scale=1)
    return (S*(X1)-(K*exp(-r*T)*X2))
print("Black Scholes Approximation",Black_Scholes(sigma,S,K,r,t))


data = pandas.read_csv('msft_15_May.csv')
A1=data['Strike']
B1=data['Bid']
B2=data['Ask']
A2=(B1+B2)/2 #OPTION
plot(A1,A2)
plt.scatter(A1,A2)
plt.title("Actual Data ")
plt.xlabel("Strike Price")
plt.ylabel("Option Price")
plt.show()



lst = []
for i in data['Strike']:
    lst.append(Black_Scholes(sigma, S, i, 1.5/100, 1/12))
print(lst)	
plot(A1,lst)
plt.scatter(A1,A2)
plt.title("Using Black Scholes")
plt.xlabel("Strike Price")
plt.ylabel("Option Price")
plt.show()

#QUESTION(d)
""""
We discussed mainly 3 methods for Option pricing in the lectures:
1. Black Scholes Method
2. Binomial Model
3. Monte Carlo Method


1.Monte Carlo Method
- best for pricing options where the payoff is path dependent
- can be used with any probability distribution i.e. not limited to normal or lognormal returns
-sometimes, varying statistical parameters allowed 


2. Black Scholes
advantages:
- Speed: calculate option price in very short time.
- high accuracy not cricical for american option, so prefered
-
disadvantages:
-not as accurate as Binomaial Model	
	

3.Binomial Model
advantages:
-Acuracy

disadvantages:
-slow	
""""
