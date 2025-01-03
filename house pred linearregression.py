import numpy as np # linear algebra
import pandas as pd 
import matplotlib.pyplot as plt 
import sys
np.set_printoptions(threshold= sys.maxsize)

#Importing DataSet 
dataset = pd.read_csv(r'C:\Users\YESHWANTH\House_data.csv')
space=dataset['sqft_living']
price=dataset['price']

x = np.array(space).reshape(-1, 1)
y = np.array(price)
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x ,y, test_size=1/3, random_state=0)

#Fitting simple linear regression to the Training Set
from sklearn.linear_model import LinearRegression 
regressor1 = LinearRegression()
regressor1.fit(xtrain, ytrain)

#Predicting the price
pred = regressor1.predict(xtest)

plt.scatter(xtrain, ytrain ,color='red')
plt.plot(xtrain, regressor1.predict(xtrain),color='blue')
plt.title("visual for training dataset")
plt.xlabel("space")
plt.ylabel("price")
plt.show()

##test
plt.scatter(xtest, ytest,color='blue')
plt.plot(xtrain, regressor1.predict(xtrain),color='red')
plt.title("visual for testing dataset")
plt.xlabel("space")
plt.ylabel("price")
plt.show()





