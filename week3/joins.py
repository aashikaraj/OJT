import numpy as np
p = np.array([1,2,3,4]) 
q = np.array([5,6,7,8])
concat_array=np.concatenate((p,q))
print("concat the array",concat_array) 
#horizental array
array_2D=np.array([[1,2,3],[4,5,6]])
array_2=np.array([[7,8,9],[10,11,12]])
hatack_array=np.hstack((array_2D,array_2))
print(hatack_array)

#vertical 
array2d_1=np.array([[1,2,3],[4,5,6]])
array2d_2=np.array([[7,8,9],[10,11,12],[13,14,15]])
v = np.vstack((array2d_1,array2d_2))
print(v)


#
vstack_array = np.vstack((array2d_1,array2d_2))
print("vertical stacked array is : ", vstack_array)


#horizental array
hstack_array = np.hstack((array2d_1,array2d_2))
print("horizental htacked array is : ", hstack_array)