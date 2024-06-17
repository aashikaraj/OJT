# coverting 1D into 2D
# reshape() is the method which is used for reshaping the arrays
import numpy as np

#create a iD Array
array_1D = np.array([1,2,3,4,5,6])
print("array1D : ",array_1D)
print("ahape pf array1D : ", array_1D.shape)

#reshape the !D array to 2D array
array_2D = array_1D.reshape((2,3))
print("array2D :",array_2D)
print("shape of array_2D : ",array_2D.shape)


#reshape the !D array to 2D array
array_3D = array_1D.reshape((3,2))
print("array_3D :",array_3D)
print("shape of array_3D : ",array_3D.shape)



#reshape back a 2d array to 1D
array_1D_back = array_2D.reshape((-1))
print("array_1D_back : ",array_1D_back)
print("shape of array_1D_back : ",array_1D_back.shape)






