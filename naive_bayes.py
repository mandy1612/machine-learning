from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix

dataset = load_iris() 
X, y = dataset.data, dataset.target 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1) 

gauss_naive_bayes = GaussianNB([0.18,0.37,0.45]) 
# training model
gauss_naive_bayes.fit(X_train, y_train) 

# testing model
y_pred = gauss_naive_bayes.predict(X_test) 


# print("Accuracy(in %):",accuracy_score(y_test, y_pred)*100)
# print("Confussion matrix:",confusion_matrix(y_test,y_pred))

# tuning of size of train test ratio is upto us
# one key observation is the trade off between accuracy and confusion matrix (aka error matrix)
# while increasing test sample size will give more accuracy, errors increases too
# whereas decreasing test sample size will make sure less errors but it affects accuracy
