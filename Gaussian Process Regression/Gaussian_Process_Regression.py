#!/usr/bin/env python
# coding: utf-8
# In[ ]:

# test
'''
Assignment 4 CS 5783 Machine Learning
@author: Rui Hu
'''
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error


# In[ ]:


data = np.loadtxt("C:/Users/Administrator/Downloads/crash.txt")
X = data[:,0]/np.amax(data[:,0])
Y = data[:,1]/np.amax(data[:,1])


# In[ ]:


def kernel_squared_exponential(x, y, delta):
    dx = np.expand_dims(x, 1) - np.expand_dims(y, 0)
    return np.exp(-1/2.0 * np.power(dx, 2) / delta**2)


# In[ ]:


def kernel_exponential(x,y,delta):
    dx = np.expand_dims(x, 1) - np.expand_dims(y, 0)
    return np.exp(-1/2.0 * np.abs(dx) / delta)


# In[ ]:


def predict(X,Y,x,delta,kernel_fc):
    belta = 2.2
    if kernel_fc == "squared_exponential":
        K = kernel_squared_exponential(X,X,delta) #
        k_n = kernel_squared_exponential(X,x,delta)
        c =  kernel_squared_exponential(x,x,delta)
    elif kernel_fc == "exponential":
        K = kernel_exponential(X,X,delta)
        k_n = kernel_exponential(X,x,delta)
        c = kernel_exponential(x,x,delta)
    
    C = K + belta*np.eye(len(X))    
    mean_y = np.dot(np.dot(k_n.T, np.linalg.inv(C)), Y)
    cov_y = c - np.dot(np.dot(k_n.T, np.linalg.inv(C)), k_n)
    exp_y = np.dot(k_n.T, np.dot(np.linalg.inv(C),Y))
    return mean_y, cov_y, exp_y


# In[ ]:


x = np.linspace(0,1,num=20)

# choose the range of delta for squared_exponential
kernel_fc = "squared_exponential"
#kernel_fc = "exponential"
Delta = np.array([0.001,0.01,0.05,0.1,0.5,1,10])
plt.figure(figsize=(30,len(Delta)))
for i in range(len(Delta)):
    delta = Delta[i]
    mean_y, cov_y, exp_y = predict(X,Y,x,delta,kernel_fc)
    y = multivariate_normal.rvs(mean_y, cov_y,size =5)
    # print(y.shape)
    plt.subplot(1,len(Delta),i+1)
    plt.plot(X,Y, 'o', color = 'r')
    for j in range(5):
        plt.plot(x,y[j,:])
    plt.xlabel("delta = %.3f" % Delta[i])

plt.show()


# In[ ]:


# choose 0.01-1
Delta = np.linspace(0.01,1, num=100)


# In[ ]:


# 5-fold
kf = cross_validation.KFold(len(X), n_folds=5)


# In[ ]:


# 
kernel_fc = "squared_exponential"
MSE = np.zeros(len(Delta))
for i in range(len(Delta)):
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        mean,cov,y_pred = predict(X_train, y_train, X_test,Delta[i],kernel_fc)
        
        MSE[i] = MSE[i]+mean_squared_error(y_pred,y_test)
plt.figure()
plt.plot(Delta,MSE,'-o')
plt.xlabel("delta")
plt.ylabel("MSE")
plt.title("Squared Exponential Kernel")
plt.show()
opt_delta = Delta[np.argmin(MSE/len(Delta))]


# In[ ]:


print("The optimal delta for squared exponential kernel is {}.".format(opt_delta))


# In[ ]:


kernel_fc = "exponential"
MSE = np.zeros(len(Delta))
for i in range(len(Delta)):
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        mean,cov,y_pred = predict(X_train, y_train, X_test,Delta[i],kernel_fc)
        
        MSE[i] = MSE[i]+mean_squared_error(y_pred,y_test)

plt.plot(Delta,MSE,'-o')
plt.xlabel("delta")
plt.ylabel("MSE")
plt.title("Exponential Kernel")
plt.show()
opt_delta = Delta[np.argmin(MSE/len(Delta))]
print("The optimal delta for squared exponential kernel is {}.".format(opt_delta))


# In[ ]:


# Inference
x = np.linspace(0,1,num=100)
delta = 0.09
kernel_fc = "squared_exponential"
#kernel_fc = "exponential"
mean_y, cov_y, pred_y = predict(X,Y,x,delta,kernel_fc)

fit = multivariate_normal.rvs(mean_y, cov_y)
plt.figure()
plt.plot(X,Y,'o',color = 'r')
plt.plot(x,pred_y,'x',color = 'b')
plt.title("Squared_exponential kernel")
plt.show()

kernel_fc = "exponential"
delta = 0.06
mean_y, cov_y, pred_y = predict(X,Y,x,delta,kernel_fc)
plt.figure()
fit = multivariate_normal.rvs(mean_y, cov_y)
plt.plot(X,Y,'o',color = 'r')
plt.plot(x,pred_y,'x',color = 'b')
plt.title("Exponential kernel")
plt.show()

