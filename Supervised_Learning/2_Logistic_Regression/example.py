import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from Logistic_Regression import LogisticRegression


bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=421)

fig = plt.figure(figsize=(8,6))
plt.scatter(X[:, 0], y, color = "r", marker= "o", s=30)
plt.show()

clf = LogisticRegression(lr=0.00001)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

def accuracy(y_test, y_pred):
    return np.sum(y_test==y_pred)/len(y_test)

accuracy = accuracy(y_test, y_pred)
print(accuracy)

 