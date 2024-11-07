import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from Linear_regression import LinearRegression


X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=421)

fig = plt.figure(figsize=(8,6))
plt.scatter(X[:, 0], y, color = "r", marker= "o", s=30)
plt.show()

reg = LinearRegression()
reg.fit(X_train, y_train)

prediction = reg.predict(X_test)

def mse(y_test, prediction):
    return np.mean((y_test-prediction)**2)

mse = mse(y_test, prediction)
print(mse)

y_predictline = reg.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, y_predictline, color= 'black', linewidth=2, label='Prediction')
plt.show()
 