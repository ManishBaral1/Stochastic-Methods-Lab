#hw10_q1

from pylab import *
from scipy.linalg import solve_banded
import timeit

""" 
GTM: not using solve_banded... -2
     since n is small not possible to make proper comparison of 
     computing times... -1
given points: 3
"""

def Tridiagonal_Solver(a1,b1,c1,d1): #Gausssian Elimination
    n = len(d1)
    p, q, r, s = map(array, (a1, b1, c1, d1))  
    for i in range(1, n):
        z = p[i-1]/q[i-1]
        q[i] = q[i] - z*r[i-1] 
        s[i] = s[i] - z*s[i-1]      	    
    x = q
    x[-1] = s[-1]/q[-1]
    for j in range(n-2, -1, -1):
        x[j] = (s[j]-r[j]*x[j+1])/q[j]
    return x

""" GTM: in efficient way of building a,b,c,d and n must be large """
a1 = array([1.,1,1]) 
b1 = array([-2.,-2.,-2.,-2.])
c1 = array([1.,1.,1.])
d1 = array([9,10,15,3.]) #Random vectors
print('Gaussian Elimination method=',Tridiagonal_Solver(a1, b1, c1, d1))
T1 = timeit.Timer('Tridiagonal_Solver', 'from __main__ import Tridiagonal_Solver,a1,b1,c1,d1')
print('100 Evaluations take ', T1.timeit(number=100), 's')

M = array([[-2,1,0,0],[1,-2,1,0],[0,1,-2,1],[0,0,1,-2]],dtype=float)   
def Build_in_solver(M,d1): #Build-in Python solver
    """GTM: you were supposed to use solve_banded function """
    return linalg.solve(M, d1)
print('Using built-in python function=',Build_in_solver(M,d1)) 
T2 = timeit.Timer('Build_in_solver', 'from __main__ import Build_in_solver,M,d1')
print('100 Evaluations take ', T2.timeit(number=100), 's')

