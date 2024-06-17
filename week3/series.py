import pandas as pd


data = [10,2,56,85,75,49,65,45,]
series = pd.Series(data)
print(series)

series_dict = pd.Series({'a':1, 'b':2, 'c':3})
print(series_dict)


 
#serires: which one dimentional.
# we can create a series with in list,array,dictionaries
#each series value have an index value which will start from 0
data = [10,2,3,45,65]
series = pd.Series(data)
 
print(series)
 
#access the element seperately by using indexing
 
print(series[3])
 
#arithematical operation
 
series_add = series + 10
 
print(series_add)
 
# filter elemnts in the series.
 
filtered_series = series[series > 20]
print(filtered_series)
 
 
#statical method
#data = [10,2,3,45,65]
 
mean = series.mean()
print (f"the mean value of the series is {mean}")