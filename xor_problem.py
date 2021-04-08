import numpy as np
from sklearn.svm import SVC
from matplotlib import pyplot as plt

# x = np.array([[0,0],[0,1],[1,0],[1,1]],dtype=np.float)
# y = np.array([0,1,1,0],dtype=np.float)

# plt.scatter(x[:,0],x[:,1])
# plt.show()

X = np.random.randn(200,2)
X_train = X[:150,:]
X_test = X[150:,:]
y_train = np.logical_xor(X_train[:,0]>0,X_train[:,1]>0)
y_test = np.logical_xor(X_test[:,0]>0,X_test[:,1]>0)

classifier = SVC()
# print(classifier.kernel)
classifier.fit(X_train,y_train)

y_predict = classifier.predict(X_test)

count = 0
for i in range(len(y_test)):
    if y_test[i] != y_predict[i]:
        count += 1

print(count)