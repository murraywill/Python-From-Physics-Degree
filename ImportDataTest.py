import numpy as np 
filepath = "/Users/jimmurray/Downloads/data_241111_009.txt"
with open(filepath, "r") as file:
    header4 = file.readline()
    header7 = file.readline()
    for line in file:
        line = line.strip()
        columns = line.split()
        xvalues = columns[5]
        yvalues = columns[8]

        xdata = float(columns[5])
        ydata = float(columns[8])
        print(xvalues, xdata)
file.close()
#f = open('data_241111_009.txt', 'r')


#header4 = f.readline()
#header7 = f.readline()

#for line in f:
    #line = line.strip()
    #columns = line.split()
    #xvalues = columns[4]
    #yvalues = columns[7]

    #xdata = float(columns[4])
    #ydata = float(columns[7])
    #print(xvalues, xdata)


