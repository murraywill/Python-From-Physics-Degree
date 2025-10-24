import numpy as np 
#Question 1 Formative
def f1(x,a,b):
    y = x**2
    y1 = np.log(b*y)
    y2 = np.cos(y1)
    y3 = np.exp(b*y2)
    y4 = 1/(np.sqrt(2*a)) * y3
    return y4 
print(f1(2,2,2))

# Question 2 

def f2(N, M, P):
    alist = np.arange(N, M + 1, P)
    return np.tan(alist)

print(f2(2,12,3))
# Question 3 

def f3(list, N):
    ar = np.array(list)
    ar1 = ar**N
    total = sum(ar1)
    return total

print(f3([3,4,5],2))

#Question 4 

def f4(a,b):
    x = a/b
    if (x == int(x)):
        return 1
    else:
        return 0
    

print(f4(2,1))

# Question 5

import matplotlib.pyplot as plt

def f5(xvalues):
    y = np.array(xvalues)
    y1 = y * np.tan(y)
    y2 = y1**2
    plt.figure(1)
    
    p = plt.plot(xvalues, y2)
    plt.show()
    return p

f5([1,2,3,4,5])
