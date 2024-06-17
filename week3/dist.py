import pandas as pd

df =pd.read_csv('Book1.csv',

dtype={'age':int, 'salary': float}, usecols=['Name', 'Age', 'Place'])

print(df)