import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import KBinsDiscretizer

# Load our data
df = pd.read_csv('housing.csv')

# Split the dataset into feature and target as (x) and (y) axis
X = df[['size', 'bedrooms']].values
y = df['price'].values

# Discretize the target variable for stratification
discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
y_binned = discretizer.fit_transform(y.reshape(-1, 1)).flatten()

# Initiate or define our model
model = LinearRegression()

# Define our cross-validation method which is Stratified KFold
skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

mae_scores = []

for train_index, test_index in skf.split(X, y_binned):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Training the model with set which gets after we get after looping
    model.fit(X_train, y_train)
    
    # Predict the test set
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mae_scores.append(mae)

average_mae = np.mean(mae_scores)
print(f"Average Mean Absolute Error (Stratified K-Fold): {average_mae}")