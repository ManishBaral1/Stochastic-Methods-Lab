#hw9.q.no3


from pylab import *
import numpy as np


"""
GTM: Since this code is focusing on only visualization:
     unnecessary for loop -1
     wrong mapping... -2
given points: 4
"""

n,r,S,X,T,σ=100,0.05,1,0.8,1,0.3
c = [[0 for x in range(n+1)] for y in range(n+1)] 


def payoff(S,X,u,d,n,i): 
    return max (0, ((S*(u**(n-i))*(d**i))-X))

def binomial_array(payoff, n, r, σ, X, S, T):
    d=1/(np.exp(σ*((T/n)**0.5)))
    p=((np.exp((r*T/n)))-d)/((np.exp(σ*((T/n)**0.5)))-d)
    for i in range(0,n+1): 
        for j in range(0,n+1):
            c[i][j]=-1
    for i in range(0,n):
        c[n][i]= payoff(S,X,(np.exp(σ*((T/n)**0.5))),d,n,i)
    for j in range(n-1,-1,-1):
        for i in range(0,j+1):
            c[j][i]=(p*c[j+1][i] + (1-p)*c[j+1][i+1])/(np.exp((r*T/n)))
    return c[0][0]

print((binomial_array(payoff, 100, 0.05, 0.3, 0.8, 1, 1)))
c=ma.masked_equal(c,-1)

imshow(c)
cbar = colorbar()
cbar.solids.set_edgecolor("face")
draw()
show()
