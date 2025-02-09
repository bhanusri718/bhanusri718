import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle

dataset=pd.read_csv(r"C:\excel\Social_Network_Ads.csv", encoding="utf-8")

X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Training the K-NN model on the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=4, p=1)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# This is to get the Models Accuracy 
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)
print(ac)

# This is to get the Classification Report
from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print(cr)

bias = classifier.score(x_train,y_train)
bias

variance = classifier.score(x_test,y_test)
variance

# Visualising the Training set results
from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() -1,stop =x_set[:,0].max() +1, step =0.01),
                     np.arange(start = x_set[:, 1].min() -1,stop =x_set[:,1].max() +1, step =0.01)
                     )

plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(['red','green']))

plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                color =['red', 'green'][i], label = j)
    
    
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
    
plt.title('K-NN (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Save the model  in binary mode
pickle_filename = "classifier.pkl"
with open(pickle_filename, "wb") as file:
    pickle.dump(classifier, file)
print(f"Model saved as {pickle_filename} in {os.getcwd()}")
