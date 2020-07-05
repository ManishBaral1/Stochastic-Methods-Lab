#HW_11

#Q_1
from pylab import *
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

"""
GTM: Problem 1)

     In this assignment, since all parts connected to each other, starting wrong
     leads other part wrong as well. But it does not mean that if I cut some grade 
     for a mistake, it should not be taken from any other parts. This is your 
     responsibility to check them properly and to notice debugs in the upcoming
     tasks.

     part a)Task: generate geom Brow, find estimates, plot semilogx, comment       (+2.5)
            missing 0 entry... -0.5
            unnecessary for loop... -1
            not updating dt -> wrong computation... -0.5
            so that, wrong plot and comment... -1.5
        
     part b)Task: geom Brown, mean and std of estimates, loglog plot, variance     (+1.5)
                  for sigma_hat, comment
            unnecessary for loops (used in two different purpose)... -1
            wrong estimates... -0.5
            the formulation for var is not computed to make some comparison
            and wrong plot and comment.... -1
        
     part c)Task: single geom Brown from (a), geom Brown as in (b), hist of        (+2.5)  
                  estimates with std of estimates and std of original model.
            code inefficiency... -1
            missing stds of estimates in the plot... -0.5      
                  
     part d)Task: changes in (a) with (1) Gaussian noise, (2) high freq. periodic  (+1)
                  part.
            due to mistake in the main func., not able to see the changes... -1
                  
     part e)Task: QQ-plot vs. normal dist of log-returns, dists of two noisy log   (+0.5)
                  returns in one graph, comment.
            the questionsn is not understood...
            0.5 is given, just for writing trying... 
     
     part f)Task: autocorr. of time-series of log-returns, and two noisy version   (+1.5) 
                  in one plot, comment.
            all autocorrelations are not in one plot, 
            no comment... -0.5
            
given points: 9.5
"""     

seed(1)
k=10
#k=20
µ,σ,K0,N= 0.2,0.4,1,2**k
dt = 1/(2**k)
t = np.linspace(0, 1, 2**k)
"""GTM: this is not equal to 0 entry at the beginning.
Write as W = r_[0, np.random.standard_normal(size = N)] """
W0 = 0+np.random.standard_normal(size = (2**k)-1) 
W0 = np.cumsum(r_[0,W0])*np.sqrt(dt)
X = (µ-0.5*σ**2)*t + σ*W0 
K = K0*np.exp(X) 

def gxn(k,K):
    a1=(K[::2**k])  
    a2= (np.log(a1[1:])-np.log(a1[:-1]))    
    samp_a1=sum(a2)/(len(a2))
    samp_σ=(1/(len(a1)))*sum((a2-samp_a1)**2)
    """ GTM: mistake in computing std"""
    #samp_σ=(1/(len(a2)-1))*sum((a2-samp_a1)**2)
    """GTM: Not updating dt"""
    #dt = 1./len(a2)
    µ_d= samp_a1/dt + 0.5* samp_σ
    σ_d=sqrt(samp_σ)/sqrt(dt)
    return σ_d,µ_d,len(a2)+1

"""GTM: unnecessary for loop, you have vectorize it"""
A1=[]
A2=[]
A3=[]
for i in range(1,k+1):
    σ, µ,mm = gxn(i,K)
    A2.append(σ)
    A1.append(µ)
    A3.append(mm)    
print(A2)

plt.semilogx(A3, A1, label='µ',color='r')
plt.semilogx(A3, A2, label='σ',color='g')
plt.title("first problem")
plt.legend()
plt.show()
"""
 =σ doesn't converge to 0.4
 =estimated values converge to true values model
"""

#Q_2 
A1mean=[]
A2mean=[]
A1std_dv=[]
A2std_dv=[]

"""GTM: Very inefficient way of creating ensemble of Brownian path 
and then computing geom. Brownian"""
Brownian_path=np.zeros((20000,N))
"""GTM: unnecessary for loop"""
for i in range(0, 20000):
    k1=10
    µ1,σ1,N1 = 0.2,0.4,2**k1
    t1 = np.linspace(0, 1, 2**k1)
    dt1=1/(2**k1)
    """GTM: Same mistake as above"""
    W1 = 0+np.random.standard_normal(size = 2**k1) 
    W1 = np.cumsum(W1)*np.sqrt(dt1) 
    X0 = (µ1-0.5*σ1**2)*t1 + σ1*W1 
    K1 = 1*np.exp(X0)
    Brownian_path[i]= K1          


A4=[]
A5=[]   

"""GTM: main idea is right but the computations are wrong
in gxn functions"""

"""GTM: unnecessary for loop"""
for j in range(1, k+1): 
    for i in range(0, 200): 
        µ2, σ2,mm2 = gxn(j,Brownian_path[i])
        A5.append(σ2)
        A4.append(µ2)
    A1mean.append(np.nanmean(A4))
    A2mean.append(np.nanmean(A5))
    A1std_dv.append(np.nanstd(A4))
    A2std_dv.append(np.nanstd(A5))

plt.loglog(A3, A1mean, label='mean-µ',color='g')
plt.loglog(A3, A2mean, label='mean-σ',color='r')
plt.loglog(A3, A1std_dv, label='sd-µ',color='y')
plt.loglog(A3, A2std_dv, label='sd-σ',color='b')
plt.legend()
plt.show()
"""
Var[\sigma^2 ] has linear order only.
"""

#Q_3

"""
Histogram (hist) plot of the estimates
for µ and σ from this ensemble, computation of the standard deviation, and visualization of the
standard deviation and the true value of the original model in this histogram.
"""

"""GTM: full of unnecessary for loops and definitions"""
A1mean2=[]
A2mean2=[]
A1std_dv2=[]
A2std_dv2=[]
g_m1=0
g_1=[]
for i in range(0,N-1):
    g_1.append(np.log(K[i+1])- np.log(K[i]))
g_v1=0
g_m1=sum(g_1)/N
for i in range(0,N):
    g_v1=g_v1+(g_1[i-1]-g_m1)**2
g_v1=g_v1/(len(g_1)-1)
g_vm1=np.sqrt(g_v1)/np.sqrt(dt)
g_mm1=g_m1/dt + g_v1**2/2 
print("sd :", g_vm1)
print("mean :", g_mm1)

µ3=g_mm1
σ3=g_vm1
A6=[]
A7=[]
for i in range(0, 200):
    """GTM: same mistakes"""
    W3 = 0+np.random.standard_normal(size = N) 
    W3 = np.cumsum(W3)*np.sqrt(dt)
    X3 = (µ3-0.5*σ3**2)*t + σ3*W3
    K3 = K0*np.exp(X3) 
    #plt.plot(t,K3)
 
    g_m3=0
    g_3=[]
    for j in range(0,N-1):
        g_3.append(np.log(K3[j+1])- np.log(K3[j]))
    g_v3=0
    g_m3=sum(g_3)/N
    for l in range(0,N-1):
        g_v3=g_v3+(g_3[l-1]-g_m3)**2
    g_v3=g_v3/(len(g_3))
    g_vm3=np.sqrt(g_v3)/np.sqrt(dt)
    g_mm3=g_m3/dt + g_v3**2/2 
    A6.append(g_mm3)
    A7.append(g_vm3)
    
"""GTM: where are stds of mu and sigma estimates?"""
plt.show()
#plt.hist(A6, bins=50,color="r")
plt.hist(A6, histtype='stepfilled')
plt.title("µ-histogram")
plt.show()
#plt.hist(A7, bins=50,color="r")
plt.hist(A7, histtype='stepfilled')
plt.title("σ-histogram")
plt.show()

#Q_4  
"""GTM: the setting is right but since the main function is 
wrong you got wrong results."""
f1=1  
K_periodic= K + f1*np.sqrt(dt)*sin(2*np.pi/50*arange(N))
K_gaussian= K + f1*np.sqrt(dt)*np.random.standard_normal(size=N)
"""plt.plot(t, K_periodic, label="noisy periodic",color="g")
plt.plot(t, K_gaussian, label="noisy gaussian", color="y")
plt.legend()
plt.show()"""

A8=[]
A9=[]
A10=[]
A11=[]
for i in range(1,k+1):
    µ_p, σ_p, mmp = gxn(i, K_periodic)
    µ_g, σ_g, mmg = gxn(i, K_gaussian)
    A8.append(µ_g)
    A10.append(σ_g)
    A9.append(µ_p)
    A11.append(σ_p)
 
plt.semilogx(A3, A1, label='µ')
plt.semilogx(A3, A2, label='σ')
plt.semilogx(A3, A8, label='µ_gaussian')
plt.semilogx(A3, A9, label='µ_periodic')
plt.semilogx(A3, A10, label='σ_gaussian')
plt.semilogx(A3, A11, label='σ_periodic')
plt.legend()
plt.show()

"""
Gaussian noise converging.

"""

#Q_5
A12=[] 
A13=[]
A14=[]

"""GTM: You need to plot normal distribution for three different
log-returns.For this, use 
r = log(S[1:]) - log(S[:-1])
nd = sort(normal(0, 1, N))
rnorm = sort((r-r.mean())/r.std())
plt.plot(nd, rnorm, color = c)"""
for i in range(0,N-1): #QQ-plot vs the normal distribution for the distribution of the log-returns, and the distribution of the two noisy log returns from Q_4
    A12.append(np.log(K[i+1])- np.log(K[i]))
    A13.append(np.log(K_gaussian[i+1])- np.log(K_gaussian[i]))
    A14.append(np.log(K_periodic[i+1])- np.log(K_periodic[i]))

stats.probplot(A12, dist="norm",plot=plt)
plt.title("QQ-first")
plt.show()
stats.probplot(A13, dist="norm",plot=plt)
plt.title("QQ-gaussian")
plt.show()
stats.probplot(A14,dist="norm",plot=plt)
plt.title("QQ-periodic")
plt.show()



#Q_6
"""GTM: All must be in one graph"""
plt.acorr(A12, color="r") #Autocorrelation for the time series of log-returns, and the two noisy versions.
plt.title("Ac-first",color="g")
plt.show()
plt.acorr(A13,color="r")
plt.title("Ac-gaussian",color="g")
plt.show()
plt.acorr(A14,color="r")
plt.title("Ac-periodic",color="g")
plt.show()

