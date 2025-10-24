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
def sample_func(x, S_i, lamda_i, std, tPrime):
    T = 13.1 * 10**3
    expon = np.exp((std**2 * lamda_i**2)/2 - (x - tPrime) * lamda_i)
    decay = (1/np.exp(lamda_i * T) - 1)
    errorfunc = scp.erf((x - tPrime) / (np.sqrt(2) * std) - (std * lamda_i)/np.sqrt(2))
    y3 = S_i * expon *( decay + (1 + errorfunc) / 2)
    return y3


def fitting_func(x, S_1, S_2, lamda_1, lamda_2, std, tPrime):
    T = 13.1 * 10**3
    return sample_func(x, S_1, lamda_1, std, tPrime) + sample_func(x, S_2, lamda_2, std, tPrime)
    
initial_guess = [1e-5, 1e-5, 1, 10, 0.15, 0]
actual_fit_parameters2, covariance_matrix2 = scpo.curve_fit(fitting_func,t_0 ,ydata_array, p0 = initial_guess)

fit_S_1 = actual_fit_parameters2[0]
fit_S_2 = actual_fit_parameters2[1]
fit_lamda_1 = actual_fit_parameters2[2]
fit_lamda_2 = actual_fit_parameters2[3]
fit_std = actual_fit_parameters2[4]
fit_tPrime = actual_fit_parameters2[5]

ybestfit = fitting_func(t_0, fit_S_1, fit_S_2, fit_lamda_1, fit_lamda_2, fit_std, fit_tPrime)



#axis at 0.5s


plt.figure(1)
plt.figure(figsize=(12, 4)) 
plt.plot(t_0, ydata_array)
plt.plot(t_0, ybestfit) 
plt.xlabel("Time/ps")
plt.ylabel("Amplitude/V")

plt.show()