import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

dataframe= pd.read_csv(r'C:\Users\YESHWANTH\Downloads\Salary_Data.csv')

x= dataframe.iloc[:,:-1].values
y=dataframe.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test, y_train,y_test =train_test_split(x, y, test_size=0.20)
    

from sklearn.linear_model import LinearRegression

# Model building
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predictions 

y_pred = regressor.predict(x_test)

# Visualization


# Visualization: Training set
plt.scatter(x_train.ravel(), y_train, color='red')  # Flatten x_train for plotting
plt.plot(x_train.ravel(), regressor.predict(x_train), color='blue')  # Regression line for training data
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualization: Test set
plt.scatter(x_test.ravel(), y_test, color='orange', label='Test Data')  # Flatten x_test for plotting
plt.plot(x_test.ravel(), regressor.predict(x_test), color='blue', label='Regression Line')  # Regression line for test data
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()


# Optional: Output the coefficients of the linear model
print(f"Intercept: {regressor.intercept_}")
print(f"Coefficient: {regressor.coef_}")

# Compare predicted and actual salaries from the test set
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison)

dataframe.mean()

dataframe['Salary'].mean()


dataframe['Salary'].mode()


dataframe.var()


dataframe.std()


dataframe['Salary'].std()

from scipy.stats import  variation

variation(dataframe.values) 

variation(dataframe['Salary']) 
 
dataframe.corr()

correlation_matrix = dataframe.corr()
print(correlation_matrix)
    
dataframe.columns

dataframe['Salary'].corr(dataframe['YearsExperience'])

dataframe['Salary'].skew()


dataframe['Salary'].sem()

import scipy.stats as stats
dataframe.apply(stats.zscore) 

stats.zscore(dataframe['Salary'])

a = dataframe.shape[0] # this will gives us no.of rows
b = dataframe.shape[1] # this will give us no.of columns
degree_of_freedom = a-b
print(degree_of_freedom)

#First we have to separate dependent and independent variables
X=dataframe.iloc[:,:-1].values #independent variable
y=dataframe.iloc[:,1].values   
# dependent variable
y_mean = np.mean(y) # this will calculate mean of dependent variable
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)
from sklearn.linear_model import LinearRegression
reg =  LinearRegression()
reg.fit(X_train,y_train)
y_predict = reg.predict(X_test) # before doing this we have to train,test and split our 
SSR = np.sum((y_predict-y_mean)**2)
print(SSR)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)
from sklearn.linear_model import LinearRegression
reg =  LinearRegression()
reg.fit(X_train,y_train)
y_predict = reg.predict(X_test) # before doing this we have to train,test and split our 
y = y[0:6]
SSE = np.sum((y-y_predict)**2)
print(SSE)

mean_total = np.mean(dataframe.values) # here df.to_numpy()will convert pandas Dataframe to Nump
SST = np.sum((dataframe.values-mean_total)**2)
print(SST)

r_square = SSR/SST
r_square
 
print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}")


from sklearn.metrics import mean_squared_error


# Predict salary for 12 and 20 years of experience using the trained model
y_15 = regressor.predict([[15]])
y_30 = regressor.predict([[30]])
print(f"Predicted salary for 15 years of experience: ${y_15[0]:,.2f}")
print(f"Predicted salary for 30 years of experience: ${y_30[0]:,.2f}")

# Check model performance
bias = regressor.score(X_train, y_train)
variance = regressor.score(X_test, y_test)
train_mse = mean_squared_error(y_train, regressor.predict(X_train))
test_mse = mean_squared_error(y_test, y_pred)

print(f"Training Score (R^2): {bias:.2f}")
print(f"Testing Score (R^2): {variance:.2f}")
print(f"Training MSE: {train_mse:.2f}")
print(f"Test MSE: {test_mse:.2f}")

import pickle
filename ='regressor.pkl'
with open(filename,'wb') as file:
     pickle.dump(regressor, file)
print("Model has been pickled and saved as linear_regression_model.pkl")

import os
print(os.getcwd())








