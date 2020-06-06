# Importing the libraries
import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

import pickle

# Reading data
df = pd.read_csv('salary_pred.csv')

# Splitting data into feature and label
X = df.iloc[:,:3] #features
y = df.iloc[:,3] #labels

# fit model with Linear regression

regressor = LinearRegression()

regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('salary_pred.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('salary_pred.pkl','rb'))
print(model.predict([[4, 9, 8]]))

