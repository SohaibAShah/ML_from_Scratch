import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from NaiveBayes import NaiveBayes


X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=123)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

clf = NaiveBayes()
clf.fit(X_train, y_train)

prediction = clf.predict(X_test)


def accuracy(y_test, y_pred):
    return np.sum(y_test==y_pred)/len(y_test)

accuracy = accuracy(y_test, prediction)
print(accuracy)
 