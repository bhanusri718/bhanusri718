# k-Fold Cross Validation

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r"C:\excel\Social_Network_Ads.csv")
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Training the Kernel SVM model on the Training set
from sklearn.svm import SVC
sv = SVC(kernel = 'rbf', random_state = 0)
sv.fit(X_train, y_train)

y_pred=sv.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
print(cm)

bias=sv.score(X_train, y_train)
bias

variance=sv.score(X_test, y_test)
variance

from sklearn.model_selection import cross_val_score
acc=cross_val_score(estimator=sv, X=X_train , y=y_train,cv=10)
print("Accuracy {:.2f}%".format(acc.mean()*100)) 
                   
       
