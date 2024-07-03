import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

#load our data
df = pd.read_csv('housing.csv')

#split the dataset into feature and target as (x) and (y)axis

x = df[['size','bedrooms']].values
y = df['price'].values

# intiate or define our model
model = LinearRegression()

# define our crossvalidation method which is kfold
kf = KFold(n_splits=5)

mae_scores = []

for train_index, test_index in kf.split(x):
    x_train , x_test = x[train_index],x[test_index]
    y_train , y_test = y[train_index],y[test_index]

    #training the model with set which gets after we gets after looping
    model.fit(x_train,y_train)

    #predict the test set
    y_pred = model.predict(x_test)

    mae = mean_absolute_error(y_test,y_pred)
    mae_scores.append(mae)

    average_mae = np.mean(mae_scores)
    print(f"average Mean Absolute error:{average_mae}")