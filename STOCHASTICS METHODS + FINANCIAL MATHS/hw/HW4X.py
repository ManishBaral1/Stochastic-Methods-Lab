
from pylab import *

"""
GTM: given point: 2
"""

def fxn(x):
    F=1000
    C=0.08
    N=10
    X=0
    for i in range (1, N+1):
        X=X+ ((C)/((1+x)**i))
    X=X + 1/((1+x)**(N))
    X=F*X
    return X

xx = linspace(0, 0.25, 25)

a=[]
for i in range(0,25):
    a.append(fxn(xx[i]))
    
    
"""
GTM: instead of appanding use this: a = fxn(xx)    
"""


plot(xx,a,'r',label = 'price of bond vs yield curve')
plt.xlabel("Yeild")
plt.ylabel("Price of bond")
plt.legend(loc='best')   
show()


