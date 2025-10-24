import numpy as np 
filepath = "/Users/jimmurray/Downloads/data_241111_009.txt"
with open(filepath, "r") as file:
    skipline1 = file.readline()
    xdata_array = []
    ydata_array = []
    for line in file:
        line = line.strip()
        columns = line.split()
        xvalues = columns[5]
        yvalues = columns[8]

        xdata = float(columns[5])
        ydata = float(columns[8])
        xdata_array = np.append(xdata_array,xdata)
        ydata_array = np.append(ydata_array,ydata)
        

        

xdata_array = np.array(xdata)
ydata_array = np.array(ydata)

file.close()

