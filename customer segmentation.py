#Imported packages and dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv("Customers.csv")
X=dataset.iloc[:,[2,9]].values

#found no. of optimal clusters

import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(X,method='ward'))
plt.title("Dendrogram")
plt.xlabel("transactions")
plt.ylabel("euclidian distances")
plt.show()

#fitted our dataset to model and made predictions

from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')

#visualised the clusters

y_hc=hc.fit_predict(X)
plt.scatter(X[y_hc==0,0],X[y_hc==0,1],s=10,c='cyan',label="1st cluster")
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],s=10,c='blue',label="2nd cluster")
#plt.scatter(X[y_hc==2,0],X[y_hc==2,1],s=10,c='red',label="3rd cluster")
plt.title("Cluster of transactions")
plt.xlabel("zip")
plt.ylabel("price")
plt.legend()
plt.show()