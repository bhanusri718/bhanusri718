import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

dataset = pd.read_csv(r"C:\Users\YESHWANTH\Downloads\Data (1).csv")
                      
x= dataset.iloc[:,:-1].values

y=dataset.iloc[:,3].values

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')

imputer=imputer.fit(x[:,1:3])

x[:, 1:3] =imputer.transform(x[:,1:3])

from sklearn.preprocessing import LabelEncoder

LabelEncoder_x =LabelEncoder()

LabelEncoder_x.fit_transform(x[:,0])

x[:,0] =LabelEncoder_x.fit_transform(x[:,0])

LabelEncoder_y = LabelEncoder()

y = LabelEncoder_y.fit_transform(y)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2, random_state=0)
