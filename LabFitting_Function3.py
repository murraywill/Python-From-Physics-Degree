import scipy.optimize as scpo
import matplotlib.pyplot as plt
import scipy.special as scp
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

file.close()


#Using Base unit of time as picoseconds
#Fitting Function
#add t' with x -t
def fitting_func(x, S_i, lamda_i, std,):
    T = 13.1 * 10**3
    expon = np.exp((std**2 * lamda_i**2)/2 - x * lamda_i)
    decay = (1/np.exp(lamda_i * T) - 1)
    errorfunc = scp.erf(x / (np.sqrt(2) * std) - (std * lamda_i)/np.sqrt(2))
    y3 = S_i * expon *( decay + (1 + errorfunc) / 2)
    return y3



actual_fit_parameters, covariance_matrix = scpo.curve_fit(fitting_func,t_0 ,ydata_array)

fit_S_i = actual_fit_parameters[0]
fit_lamda_i = actual_fit_parameters[1]
fit_std = actual_fit_parameters[2]

def fitting_func2(x, std):
    T = 13.1 * 10**3
    S_i = fit_S_i
    lamda_i = fit_lamda_i
    expon = np.exp((std**2 * lamda_i**2)/2 - x * lamda_i)
    decay = (1/np.exp(lamda_i * T) - 1)
    errorfunc = scp.erf(x / (np.sqrt(2) * std) - (std * lamda_i)/np.sqrt(2))
    y3 = S_i * expon *( decay + (1 + errorfunc) / 2)
    return y3

actual_fit_parameters2, covariance_matrix2 = scpo.curve_fit(fitting_func2,t_0 ,ydata_array)

fit_std2 = actual_fit_parameters2[0]


ybestfit = fitting_func2(t_0, fit_std2)


#axis at 0.5s


plt.figure(1)
plt.figure(figsize=(12, 4)) 
plt.plot(t_0, ydata_array)
plt.plot(t_0, ybestfit) 
plt.xlabel("Time/ps")
plt.ylabel("Amplitude/V")

plt.show()