import numpy as np

array = np.array([10,11,12,13,14,15,16,17])
#np.where(array == 20)
#where( ): use to check the particular condition for fileter and conditions

#element greater then 15 for the above array

elements = np.where(array>15, 0, array)
#print(array[elements]) 
print(elements)
#print(array)
