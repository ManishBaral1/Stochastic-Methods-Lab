

from pylab import *


"""
GTM: - wrong formulation
     - another switched bond rate and interest rate
     (-1)

given points 2
"""


def Price_validity(N,c,F,x):
    m=2
    F=1000
    #c=0.10
    C=F*c/m
    X=0
    #y = x/m
    """
    GTM: be careful while writing the function...
    """
    #X= X- ((C/x)*N - (C/(x**2))*(((1+x)**(N + 1)) - (1+x)) - N*F)/((C/x)*((1+x)**(N + 1)-(1+x)) + F*(1+x))
    y = x/m
    z1 = (C/y)*(n*m)-(C/y**2)*((1+y)**(n*m+1)-(1+y))-n*m*F
    z2 = (C/y)*((1+y)**(n*m+1)-(1+y))+F*(1+y)
    X = -z1/z2
    return X

xx = linspace(0, 100, 100)
a=[]
b=[]
c=[]
n=1
for i in range(0,100):
    a.append(Price_validity(n,0.02,1000,0.06))
    b.append(Price_validity(n,0.06,1000,0.06))
    c.append(Price_validity(n,0.12,1000,0.06))
    n=n+1

plt.plot(xx, a, 'g', label = '2% coupon rates') 
plt.plot(xx, b, 'b', label='6% coupon rates') 
plt.plot(xx, c, 'r', label='12% coupon rates') 
"""GTM: since you are asked time to maturity, reverse the x axis.... """
plt.xlim(100,0)
plt.xlabel("Time to maturity")
plt.ylabel("Price volatility")
plt.legend(loc='best')
plt.gca()
plt.show()



