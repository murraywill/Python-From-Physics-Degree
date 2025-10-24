import numpy as np
import math 

def f1(t):
    x = np.cos(2.5*t)
    s = 9 - 4.8 * t - 6.62 * x
    return s
print(f1(1))

def f2(a,b):
    list = []
    x = a + 2*b
    y = a - 2*b 
    list.append(x)
    list.append(y)
    return list
print(f2(5,1))

def f3(x,N):
    expon = np.exp(x)
    list = []
    for i in range(0,N+1):
        j = x**i / (math.factorial(i))
        list.append(j)
    y = sum(list)
    return abs(expon - y)

print(f3(5,1))

#Question 4

def f4(x,y,z):
    a = x*(y - (z**2))
    b = z * ((y**2) - x)
    c = y * (z - (x**2))
    jist = [a,b,c]
    return jist

print(f4(1,2,3))

# Question 5 

def f5(a, m, k):
    value = a
    for i in range(0,m):
        value = np.sin(value * k)
    return value

print(f5(2,2,1))
