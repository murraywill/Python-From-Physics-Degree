import scipy.optimize as scpo
import matplotlib.pyplot as plt
import sympy as sp
import numpy as np 
import math
#Getting Data

filepath = "/Users/jimmurray/Downloads/data_241111_009.txt"
with open(filepath, "r") as file:
    skipline1 = file.readline()
    t_0 = []
    ydata_array = []
    for line in file:
        line = line.strip()
        columns = line.split()
        xvalues = columns[5]
        yvalues = columns[8]

        xdata = float(columns[5])
        ydata = float(columns[8])
        t_0 = np.append(t_0,xdata)
        ydata_array = np.append(ydata_array,ydata)

#Data


file.close()


#Using Base unit of time as picoseconds
#Fitting Function
def fitting_func(x, S_i, lamda_i, std):
    T = 13.1 * 10**3
    y1 = S_i * sp.exp((std**2 * lamda_i**2)/2 - x * lamda_i) * ((1/sp.exp(lamda_i * T) - 1) + (1 / 2))
    y2 = S_i * math.exp((std**2 * lamda_i**2)/2 - x * lamda_i) * (math.erf(x / (sp.sqrt(2) * std) - (std * lamda_i)/math.sqrt(2))/2)
    y3 = y1 + y2
    return y3

print(fitting_func(t_0,1,1,1))

actual_fit_parameters, covariance_matrix = scpo.curve_fit(fitting_func,t_0 ,ydata_array)

fit_S_i = actual_fit_parameters[0]
fit_lamda_i = actual_fit_parameters[1]
fit_std = actual_fit_parameters[2]

ybestfit = fitting_func(t_0, fit_S_i, fit_lamda_i, fit_t_0, fit_std)

plt.figure(1)

plt.plot(t_0, ydata_array)

plt.plot(t_0, ybestfit) 
plt.xlabel("x values")
plt.ylabel("y values")

plt.show()
