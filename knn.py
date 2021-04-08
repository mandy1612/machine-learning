from sklearn.datasets import load_iris 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

dataset = load_iris()
X,y = dataset.data, dataset.target
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.2,random_state=1)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)

kmeans = KMeans(n_clusters=3,random_state=1)
kmeans.fit(X_train)

y_predict_knn = knn.predict(X_test)
y_predict_kmeans = kmeans.predict(X_test)
# loss = np.abs(y_test - y_predict)
# print(loss)
print(y_test[:10],"\n",y_predict_knn[:10],"\n",y_predict_kmeans[:10])