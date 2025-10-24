# Question 1
import numpy as np
def f1(data):
    total = 0
    length = len(data)
    for i in range(0,(length)):
        x = data[i]
        total = total + x
    mean = total/len(data)
    std_total = 0
    for j in range(0,(length)):
        a = (data[j]-mean) ** 2
        std_total = std_total + a

    std = np.sqrt(std_total / (length-1))
    stderr = std / np.sqrt(length - 1)
    list1 = [mean, stderr]
    array = np.array(list1)
    return array
print(f1([1,2,4]))




    

