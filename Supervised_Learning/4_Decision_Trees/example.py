import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from DecisionTree import DecisionTree


bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=421)

clf = DecisionTree(max_depth=5)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

def accuracy(y_test, y_pred):
    return np.sum(y_test==y_pred)/len(y_test)

accuracy = accuracy(y_test, y_pred)
print(accuracy)

 