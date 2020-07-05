

from pylab import *
import matplotlib.pyplot as plt


"""
GTM: You should recode the function fxn again
given points: 1
"""

""" GTM: you have mixed the coupon rate and interest rate """
def fxn(n,c,F,x):
    m=2
    k=0
    for i in range (1, n*m+1):
        k=k+ ((c/2)/((1+x/2)**i))
    #k = (c/m)/((1+x/2)**)
    k=k + 1/((1+x/2)**(n*2))
    k=F*k
    return k

""" GTM: 200 years were very long so lets restrict it to 10 years and
since it is paid semiannually, compute it at every half years...
And as it is paid semiannually, you should calculate it twice in a year.. """
xx = linspace(0, 20, 20+1)
a=[]
b=[]
c=[]

n=0
for i in range(0,21):
    a.append(fxn(n,0.02,1000,0.06))
    b.append(fxn(n,0.06,1000,0.06))
    c.append(fxn(n,0.12,1000,0.06))
    n=n + 1

plt.gca()
plt.plot(xx, a, 'r', label = '2% coupon rate') 
plt.plot(xx, b, 'b',label = '6% coupon rate') 
plt.plot(xx, c, 'g', label = '10% coupon rate') 
plt.xlabel("Time to maturity for level coupon bond")
plt.ylabel("Price")

"""GTM: since you are asked time to maturity, 
reverse the x axis.... """
xlim(10,0)
plt.legend(loc='best')
plt.show()

