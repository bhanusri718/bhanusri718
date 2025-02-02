import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset =pd.read_csv(r"C:\excel\Social_Network_Ads.csv")

x = dataset.iloc[:,2: 4].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y)

# Training the Kernel SVM model on the Training set
from sklearn.svm import SVC
sv=SVC()
sv.fit(x_train,y_train)

y_pred=sv.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac =accuracy_score(y_test, y_pred)
print(ac)

from sklearn.model_selection import cross_val_score
acc = cross_val_score(estimator=sv, X =x_train,y=y_train)
print("Accuracy: {:.2f} %".format(acc.mean()*100))
print("Standard Deviation: {:.2f} %".format(acc.std()*100))

from sklearn.model_selection import GridSearchCV
parameters =[{'C': [1, 10,100],'kernel':['linear']},
             {'C':[1,10,100], 'kernel':['rbf'],'gamma':[00.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}]
grids = GridSearchCV(estimator=sv , param_grid = parameters , scoring ='accuracy',cv=10)

grids = grids.fit(x_train,y_train)
best_accuracy = grids.best_score_
best_parameters = grids.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)










