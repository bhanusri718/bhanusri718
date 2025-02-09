import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\excel\Mall_Customers.csv")

x=data.iloc[:,[3,4]].dropna().values

from sklearn.cluster import KMeans
Wcss=[]
print(len(Wcss)) 

for i in range(1,11):
    print(f"Training k-means with {i} clusters...")
    kmeans=KMeans(n_clusters=i, init='k-means++', random_state=0)
    kmeans.fit(x)
    Wcss.append(kmeans.inertia_)
plt.plot(range(1,11),Wcss)
plt.title('the elbow method')
plt.xlabel('no of clusters')
plt.show() 

#wcss we have very good parameter called inertia_ credit goes to sklearn , that computes the sum of square , formula it will compute
# Training the K-Means model on the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(x)

 ##visualize the clusters
plt.scatter(x[y_kmeans==0,0], x[y_kmeans==0,1], s=100, c='blue', label='cluster 1')
plt.scatter(x[y_kmeans==1,0], x[y_kmeans==1,1], s=100, c='green', label ='cluster 2')
plt.scatter(x[y_kmeans==2,0], x[y_kmeans==2,1], s=100, c='red', label='cluster 3')
plt.scatter(x[y_kmeans==3,0], x[y_kmeans==3,1], s=100, c='purple' ,label='cluster 4')
plt.scatter(x[y_kmeans==4,0], x[y_kmeans==4,1], s=100, c='magenta', label='cluster 5') 

# Centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('clusters of customers')
plt.xlabel('Annual income')
plt.ylabel("spending score") 
plt.legend()   
plt.show()
    
import pickle 
filename='kmeans_model.pkl'
with open(filename,'rb') as file:
    model = pickle.load(file) 
    print("model has been pickeled and saved as kmeans_model.pkl")
    
import os 
print(os.getcwd())    
