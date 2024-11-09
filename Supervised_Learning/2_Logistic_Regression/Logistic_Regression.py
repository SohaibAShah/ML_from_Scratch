import numpy as np

def sigmoid(x):
  """Calculates the sigmoid function."""
  return 1 / (1 + np.exp(-x))

class LogisticRegression:

  def __init__(self, lr=0.001, n_iters=1000):
    """
    Initializes the LogisticRegression class with learning rate (lr) and number of iterations (n_iters).

    Args:
      lr (float, optional): Learning rate. Defaults to 0.001.
      n_iters (int, optional): Number of iterations for training. Defaults to 1000.
    """
    self.lr = lr
    self.n_iters = n_iters
    self.weights = None
    self.bias = None

  def fit(self, X, y):
    """
    Fits the logistic regression model to the training data.

    Args:
      X (numpy.ndarray): Training data features.
      y (numpy.ndarray): Training data labels.
    """
    n_samples, n_features = X.shape
    self.weights = np.zeros(n_features)
    self.bias = 0

    for _ in range(self.n_iters):
      linear_pred = np.dot(X, self.weights) + self.bias
      predictions = sigmoid(linear_pred)

      dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
      db = (1 / n_samples) * np.sum(predictions - y)

      self.weights = self.weights - self.lr * dw
      self.bias = self.bias - self.lr * db

  def predict(self, X):
    """
    Predicts class labels for new data points.

    Args:
      X (numpy.ndarray): New data points for prediction.

    Returns:
      numpy.ndarray: Predicted class labels.
    """
    linear_pred = np.dot(X, self.weights) + self.bias
    y_pred = sigmoid(linear_pred)
    class_pred = [0 if y <= 0.5 else 1 for y in y_pred]
    return class_pred