import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from KNearestNeighbors import KNN

cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

iris = datasets.load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=421)

fig = plt.figure()
plt.scatter(X[:, 2], X[:,3], c=y, cmap=cmap, edgecolors= "k", s=20)
plt.show()

clf = KNN(k=5)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(y_pred)
def accuracy(y_test, y_pred):
    return np.sum(y_test==y_pred)/len(y_test)

accuracy = accuracy(y_test, y_pred)
print(accuracy)

 