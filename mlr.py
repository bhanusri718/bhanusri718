import pandas as pd

import matplotlib.pyplot as plt

import numpy as np
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv(r'C:\Users\YESHWANTH\Downloads\Investment.csv')

x =data .iloc[:,:-1]

y =data .iloc[:, 4]

# Encode or drop non-numeric columns
for col in x.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    x[col] = le.fit_transform(x[col])
    

from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test =train_test_split(x, y, test_size=0.20, random_state= 0)

from sklearn.linear_model import LinearRegression

multiregressor=LinearRegression()
multiregressor.fit(x_train,y_train)

y_pred = multiregressor.predict(x_test)

m = multiregressor.coef_
print(m)

c = multiregressor.intercept_
print(c)

x = np.append(arr = np.ones((50,1)).astype(int), values = x, axis = 1 ) 

import statsmodels.api as sm
x_opt = x[:,:x.shape[1]]
#OrdinaryLeastSquares
multiregressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
multiregressor_OLS.summary()

import statsmodels.api as sm
x_opt = x[:,[0,1,2,3]]
#OrdinaryLeastSquares
multiregressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
multiregressor_OLS.summary()


import statsmodels.api as sm
x_opt = x[:,[0,1,3]]
#OrdinaryLeastSquares
multiregressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
multiregressor_OLS.summary()


import statsmodels.api as sm
x_opt = x[:,[0,1]]
#OrdinaryLeastSquares
multiregressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
multiregressor_OLS.summary()

bias = multiregressor.score(x_train, y_train)
bias

variance = multiregressor.score(x_test, y_test)
variance
