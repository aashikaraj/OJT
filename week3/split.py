import numpy as np

#split will work numpy
array = np.array([1,2,3,4,5,6,7,8,9])
split_array = np.split(array, 3)
print("origenal array : ",array)
print("split array: ",split_array)


#multi dimentions
#horzintally and vertically


array_2d = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
vsplit_array = np.vsplit(array_2d,2)
print("vsplited array: ",vsplit_array)


#horzintally and vertically


array_2d = np.array([1,2,3,4],[5,6,7,8])
vsplit_array = np.hsplit(array_2d,2)
print("vsplited array: ",vsplit_array)