#dataframes is two dimentional which is in a tabular format.
 
import pandas as pd
 
#create a dataframe with a dictionary
data = {
    'Name': ['kigini','tuttu','ikka'],
    'Age': [24,23,24],
    'Place':['koovode','mavoor','punoor']
}
 
#conevrt the data into dataframes
df = pd.DataFrame(data)
 
 
print(df)
print(df[['Name','Place']])
 
#for accessing each row rom the dataframe we need to use the inbuilt function in pandas, iloc()
print(df.iloc[2])
 
print(df[df['Age'] > 23])