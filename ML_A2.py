#!/usr/bin/env python
# coding: utf-8

# In[116]:


import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[117]:


class LogisticRegression:
  def __init__(self, max_epochs=100):
    self.max_epochs = max_epochs
    self.weights= None
    self.bias= None

  def Calculate_Soft(self, z):
    Calculate_Soft = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
    return Calculate_Soft

  def onehotencoder(self, y):
    # Perform one hot encoding
    target_encoded = np.zeros((len(y), len(np.unique(y))))
    for i in range(len(y)):
        target_encoded[i, y[i]] = 1   
    return target_encoded

  def predict(self,X):
      z = np.dot(X, self.weights) + self.bias
      y_predi = self.Calculate_Soft(z)
      y_predi = np.argmax(y_predi,axis=1)
      return y_predi

  def acc(self,y_pred,y_test):
    for i in range(len(y_pred)):
      a= abs(y_pred[i]-y_test[i])
    return a

  def fit(self, X, y,alpha=.1, max_epochs=1000):
    # Number of samples and features
    n_sample, n_feautures = X.shape
    n_output= len(np.unique(y))
    
    # Initialize the weights and bias based on the shape of X and y.
    self.weights = np.random.randn(n_feautures,n_output)
    self.bias = np.random.randn(1,n_output)

    # Performing one hot encoding on target variable of dataset
    Ytrue = self.onehotencoder(y)

    for epoch in range(max_epochs):
      # Training the model
      z = np.dot(X, self.weights) + self.bias
      y_predi = self.Calculate_Soft(z)
      derivative_weights = (1/n_sample) * np.dot(X.T, ((y_predi - Ytrue)))
      derivative_bias = (1/n_sample) * np.sum(((y_predi - Ytrue)), axis=0)
      self.weights -= alpha * derivative_weights
      self.bias -= alpha * derivative_bias


# In[118]:


# Load the Iris dataset
iris_data = load_iris()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.1, stratify=iris_data.target)

# Object for Logistic Regression
Logi_Reg= LogisticRegression()

# Fitting the training data
Logi_Reg.fit(X_train,y_train)

# Predicting values for testing data
Ypred = Logi_Reg.predict(X_test)
print(Ypred,y_test)

# acc of prediction
a = Logi_Reg.acc(Ypred,y_test)
print (a)


# In[119]:


# Initializing Classifiers
lr = LogisticRegression()

# Loading some example data
X, y = iris_data.data, iris_data.target
X = X[:,[0, 1]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=32)

# Plotting Decision Regions
fig = plt.figure(figsize=(10, 8))

lr.fit(X_train, y_train)
fig = plot_decision_regions(X=X_test, y=y_test, clf=lr, legend=2)
plt.title("Sepal Length/ Sepal Width")
plt.show()


# In[120]:


# Initializing Classifiers
lr = LogisticRegression()

# Loading some example data
X, y = iris_data.data, iris_data.target
X = X[:,[2, 3]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=32)

# Plotting Decision Regions
fig = plt.figure(figsize=(10, 8))

lr.fit(X_train, y_train)
fig = plot_decision_regions(X=X_test, y=y_test, clf=lr, legend=2)
plt.title("Petal Length/ Petal Width")
plt.show()


# In[121]:


import numpy as np

class LDA:
    def __init__(self):
        self.pi = None     # prior probabilities
        self.mu = None     # class means
        self.sigma = None  # shared covariance matrix
    
    def fit(self, X, y):
        n, d = X.shape    # number of samples and features
        k = len(np.unique(y))  # number of classes
        self.pi = np.zeros(k)
        self.mu = np.zeros((k, d))
        self.sigma = np.zeros((d, d))
        
        for i in range(k):
            X_i = X[y == i, :]
            n_i = len(X_i)
            self.pi[i] = n_i / n
            self.mu[i, :] = np.mean(X_i, axis=0)
            self.sigma += (n_i - 1) * np.cov(X_i, rowvar=False)
        
        self.sigma /= (n - k)
        
    def predict(self, X):
        llogistic_Lik_hood = np.zeros((X.shape[0], len(self.pi)))
        for i in range(len(self.pi)):
            llogistic_Lik_hood[:, i] = np.log(self.pi[i])                                    - 0.5 * np.log(np.linalg.det(self.sigma))                                    - 0.5 * np.sum((X - self.mu[i, :]) @ np.linalg.inv(self.sigma) * (X - self.mu[i, :]), axis=1)
        
        return np.argmax(llogistic_Lik_hood, axis=1)


# In[122]:


lda = LDA()

# Fitting the training data
lda.fit(X_train,y_train)

# Predicting values for testing data
Ypred = lda.predict(X_test)
print(Ypred,y_test)


# In[123]:


# Initializing Classifiers
lda = LDA()

# Loading some example data
X, y = iris_data.data, iris_data.target
X = X[:,[0, 1]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=32)

# Plotting Decision Regions
fig = plt.figure(figsize=(10, 8))

lda.fit(X_train, y_train)
fig = plot_decision_regions(X=X_test, y=y_test, clf=lda, legend=2)
plt.title("Sepal Length/ Sepal Width")
plt.show()


# In[124]:


# Initializing Classifiers
lda = LDA()

# Loading some example data
X, y = iris_data.data, iris_data.target
X = X[:,[2, 3]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=32)

# Plotting Decision Regions
fig = plt.figure(figsize=(10, 8))

lda.fit(X_train, y_train)
fig = plot_decision_regions(X=X_test, y=y_test, clf=lda, legend=2)
plt.title("Petal Length/ Petal Width")
plt.show()

