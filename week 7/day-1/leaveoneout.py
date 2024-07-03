import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load our data
df = pd.read_csv('housing.csv')

# Split the dataset into feature and target as (x) and (y) axis
X = df[['size', 'bedrooms']].values
y = df['price'].values

# Initiate or define our model
model = LinearRegression()

# Define our cross-validation method which is LeaveOneOut
loo = LeaveOneOut()

mae_scores = []

for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Training the model with set which gets after we get after looping
    model.fit(X_train, y_train)
    
    # Predict the test set
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mae_scores.append(mae)

average_mae = np.mean(mae_scores)
print(f"Average Mean Absolute Error (Leave-One-Out): {average_mae}")