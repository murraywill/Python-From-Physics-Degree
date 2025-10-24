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


def fitting_func(x, S_1, S_2, S_3, S_4, lamda_1, lamda_2, lamda_3,lamda_4, std, tPrime, offset):
    T = 13.1 * 10**3
    return offset + sample_func(x, S_1, lamda_1, std, tPrime) + sample_func(x, S_2, lamda_2, std, tPrime) + sample_func(x,S_3, lamda_3, std, tPrime) + sample_func(x,S_4, lamda_4, std, tPrime)
    
initial_guess = [1e-5, 1e-5, 1e-5,1e-5, 1, 0.01, 0.1, 0.75, 0.15, 0, 3e-6]
actual_fit_parameters2, covariance_matrix2 = scpo.curve_fit(fitting_func,t_0 ,ydata_array, p0 = initial_guess ,maxfev = 100000)
perr = np.sqrt(np.diag(covariance_matrix2))
print('S_1 error:', perr[1],', S_2 error:', perr[2],', S_3 error:', perr[3],', lamda_1 error:', perr[4],', lamda_2 error:', perr[5], ', lamda_3 error:', perr[6], ', lamda_4 error: ', perr[7], ', std error:', perr[8], ', Tprime error:', perr[9], ', offset error:', perr[10])



#UnWeighted Data Fit
fit_S_1 = actual_fit_parameters2[0]
fit_S_2 = actual_fit_parameters2[1]
fit_S_3 = actual_fit_parameters2[2]
fit_S_4 = actual_fit_parameters2[3]
fit_lamda_1 = actual_fit_parameters2[4]
fit_lamda_2 = actual_fit_parameters2[5]
fit_lamda_3 = actual_fit_parameters2[6]
fit_lamda_4 = actual_fit_parameters2[7]
fit_std = actual_fit_parameters2[8]
fit_tPrime = actual_fit_parameters2[9]
fit_offset = actual_fit_parameters2[10]

ybestfit = fitting_func(t_0, fit_S_1, fit_S_2, fit_S_3, fit_S_4, fit_lamda_1, fit_lamda_2, fit_lamda_3, fit_lamda_4, fit_std, fit_tPrime,fit_offset)


#Weighted Data Fit
final_guess = [fit_S_1, fit_S_2, fit_S_3, fit_S_4, fit_lamda_1, fit_lamda_2, fit_lamda_3, fit_lamda_4, fit_std, fit_tPrime, fit_offset]
popt2, pcov2 = scpo.curve_fit(fitting_func, t_0, ydata_array, p0 = final_guess, sigma = (np.sqrt(ydata_array)), absolute_sigma=True, maxfev = 100000)

#Weighted Data Fit 
Wfit_S_1 = popt2[0]
Wfit_S_2 = popt2[1]
Wfit_S_3 = popt2[2]
Wfit_S_4 = popt2[3]
Wfit_lamda_1 = popt2[4]
Wfit_lamda_2 = popt2[5]
Wfit_lamda_3 = popt2[6]
Wfit_lamda_4 = popt2[7]
Wfit_std = popt2[8]
Wfit_tPrime = popt2[9]
Wfit_offset = popt2[10]

yWbestfit = fitting_func(t_0, Wfit_S_1, Wfit_S_2, Wfit_S_3, Wfit_S_4, Wfit_lamda_1, Wfit_lamda_2, Wfit_lamda_3, Wfit_lamda_4, Wfit_std, Wfit_tPrime, Wfit_offset)
#axis at 0.5s
#Making Broken Axis

lineardata = 0.5
index = np.argmax(t_0 >= lineardata)
t_0Linear = t_0[:index]
t_0Log = t_0[index+1:]
ybestfitLinear = ybestfit[:index]
ybestfitLog = ybestfit[index+1:]
ydata_arrayLinear = ydata_array[:index]
ydata_arrayLog = ydata_array[index+1:]

yWbestfitLinear = yWbestfit[:index]
yWbestfitLog = yWbestfit[index+1:]



fig, (ax1, ax2) = plt.subplots(1, 2,gridspec_kw={'width_ratios': [1, 3]},figsize=(12,5))

fig.subplots_adjust(wspace=0.03)
ax1.plot(t_0Linear, ydata_arrayLinear)
ax1.plot(t_0Linear, ybestfitLinear)
ax1.plot(t_0Linear, yWbestfitLinear)
ax2.plot(t_0Log, ydata_arrayLog)
ax2.plot(t_0Log, ybestfitLog)
ax2.plot(t_0Log, yWbestfitLog)
ax1.set_xlim(t_0[0], t_0[index]) 
ax2.set_xlim(t_0[index+1], 1600)
ax1.set_ylim(-1e-6,1.5e-5)
ax2.set_ylim(-1e-6,1.5e-5)
ax1.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.yaxis.tick_right()
ax2.tick_params(labelleft=False)
ax2.set_xscale('log') 
ax1.set_xscale('linear')



plt.figure(1)
plt.figure(figsize=(12, 4)) 
plt.plot(t_0, ydata_array)
plt.plot(t_0, ybestfit) 
plt.xlabel("Time/ps")
plt.ylabel("Amplitude/V")

plt.show()

