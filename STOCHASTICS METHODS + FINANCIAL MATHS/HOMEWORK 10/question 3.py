#HW10_Q3

import numpy as np
from scipy.stats import norm, linregress
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
from mpl_toolkits.mplot3d import axes3d

"""
GTM: Each part is 3 points.
     part a) task: explicit Euler + boundary condition. (+2)
          wrong BCs...-0.5
          unnecessary for loop.. -0.5
     part b) task: explicit scheme being unstable for large Dt (+0.5)
          again for loop, wrong plotting... -2
          unnecessary for loop.. -0.5
     part c) task: implicit Euler + stability for large Dt, (+1.5)
          BC mistake in implicit Euler... -0.5
          plotting error... -1
     part d) task: convergence of implicit scheme when M=N (+0)
          missing... -3
given points: 4
"""

#..........................................................................
def PAYOFF(S, X):
    return np.fmax(0.0, S-X)

def BOUNDARY_CONDITIONS(dt, dX, T, Xmax, K=0.7):
    """GTM: first choose number of time and space steps and T, Xmax
    then compute dx and dt."""
    M = np.zeros((int(Xmax / dX), int(T / dt)))
    """GTM: no need this line"""
    M[0:]  = 0                  
    """GTM: Wrong bounday, it must be
    M(t, Smax) = S-K*exp(-r*dt)"""
    M[-1:] = np.exp(Xmax) - K 
    """GTM: do it without using for loop"""
    for i in range(int(Xmax / dX)):
        M[i, -1] = PAYOFF(np.exp(i*dX), K)     
    return M

def EXPLICIT_METHOD(dt, dX, T, Xmax, K=0.7, σ=0.4, r=0.03):
    M = BOUNDARY_CONDITIONS(dt, dX, T, Xmax, K)   
    """
    GTM: see the following code and fill the gaps:
    a = dt* (sigma**2/(2*dx**2) - (r-sigma**2/2)/(2*dx))
    b = 1 - dt * (sigma**2/dx**2 + r)
    c = dt * (sigma**2/(2*dx**2) + (r-sigma**2/2)/(2*dx))
    VV = np.full(M+1, 0.0)
    for m in range(M,0,-1):
        VV[1:N] = a*V[0:N-1] + b*V[1:N] + c*V[2:N+1]
        V = VV[:] """
   
    """GTM: unnecessary for loop. This way code is very slow"""
    for j in range(int(T/dt)-2, -1, -1):
        for k in range(int(Xmax/dX)-1):    
            M[k][j] = M[k][j+1] + dt * (
            σ**2/2*(M[k-1][j+1] - 2*M[k][j+1] + M[k+1][j+1])/dX**2
            + (r-σ**2/2)*(M[k+1][j+1] - M[k-1][j+1])/(2*dX)
            - r*M[k][j+1])
    M = M.transpose()
    C = M[0]         
    return C

#...........................................................................
def BLACK_SCHOLES(S=1, K=0.7, σ=0.4, r=0.03, T=1):
    P = (np.log(S/K) + (r+σ**2/2)*T) / (σ*T**0.5)
    C = S*norm.cdf(P) - K*np.exp(-r*T) * norm.cdf(P-σ*T**0.5)
    return C

"""GTM: very small number of steps in space"""
N = 20
T,Xmax = 1,N
DelT = np.linspace(T/3, T/100, N)
DelX = np.linspace(Xmax/3, Xmax/100, N)
err = np.empty((len(DelX), len(DelT))) 
"""GTM: unnecessary for loops. Vectorize the code.
But the idea is right up to some point"""
for k in range(len(DelX)):
    for j in range(len(DelT)):
        K1 = EXPLICIT_METHOD(DelT[j], DelX[k], T, Xmax)
        K2 = BLACK_SCHOLES(np.exp(np.arange(0, Xmax, DelX[k])))
        #print(k, DelX[k])        
        #print(K1, K2)
        err[k][j] = abs(K1[0] - K2[0])
        
"""GTM: To check stability, choose a point in C, and
plot it as a function of number of steps. Then, one can observe
fluctations in value for the same point. In your plots, the fluctuation
is not visible. There is no need to take logarithmic value.
If you remember from earlier exercises, even in loglog plot we do not 
take logarithm of parameters to plot.""" 
#err = np.log(err)
fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')
ax.contourf(DelT, DelX, err.transpose(), cmap=mcm.Blues)
plt.title("Unstable ")
ax.set_zlabel("error log (delta C)")
plt.show()

#..........................................................................
def TRIDIAGONAL_SOLVER(a1, b1, c1, d1):
    n = len(b1)
    bt = [b1[0]] 
    ct = [c1[0]]
    dt = [d1[0]]
    for i in range(1, n):
        bt.append(bt[i-1]*b1[i] - a1[i-1]*ct[i-1])
        dt.append(bt[i-1]*d1[i] - a1[i-1]*dt[i-1])
        if i < n-1:
            ct.append(bt[i-1]*c1[i])
    x = [dt[n-1] / bt[n-1]]
    for i in range(n-2, -1, -1):
        x.append((dt[i] - ct[i]*x[n-2-i])/bt[i])
    return list(reversed(x))



def IMPLICIT_METHOD(dt, dX, T, Xmax, K=0.7, σ=0.4, r=0.03):
    V = BOUNDARY_CONDITIONS(dt, dX, T, Xmax, K).transpose() 
    n = int(Xmax/dX)
    """GTM: Nicely written but still there is a problem of BCs"""
    a1 = np.ones(n-1) * dt / (2*dX) *(r-σ**2/2 - σ**2/dX)
    b1 = np.ones(n) * (1 + dt * σ**2 / dX**2 + dt * r)    
    c1 = np.ones(n-1) * dt / (2*dX) *(-r+σ**2/2 - σ**2/dX)  
    for t in range(int(T/dt)-2, -1, -1):
        d1 = V[t+1]
        V[t] = TRIDIAGONAL_SOLVER(a1, b1, c1, d1)    
    return V[0]

"""GTM: same comment as above"""
DelT = np.linspace(T/10, T/100, N)
DelX = np.linspace(Xmax/10, Xmax/100, N)
err = np.empty((len(DelX), len(DelT)))
for k in range(len(DelX)):
    for j in range(len(DelT)):
        K1 = IMPLICIT_METHOD(DelT[j], DelX[k], T, Xmax)
        K2 = BLACK_SCHOLES(1) 
        err[k][j] = abs(K1[0] - K2)

fig2 = plt.figure()
ax = fig2.add_subplot(111, projection='3d')
ax.contourf(DelT, DelX, err.transpose(), cmap=mcm.Greens)
plt.title("Stable ")
ax.set_zlabel("error (delta C)")
plt.show()

