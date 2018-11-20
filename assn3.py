
'''
Assignment 3 CS 5783 Machine Learning
@author: Rui Hu
'''
import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import minimize


#
def flower_to_float(s):
	d = {b"Iris-setosa":0,b"Iris-versicolor":1,b"Iris-virginica":2}
	return d[s]
irises = np.loadtxt("/Users/hurui/Downloads/CS 5783 ML/iris.data.txt", delimiter=',',converters={4:flower_to_float})
X_iri = irises[:,:-1]
Y_iri_raw = irises[:,-1]
# one-hot array
Y = np.array(Y_iri_raw, dtype = np.int)
Y_iri = np.eye(3)[Y]
#separate
mask = np.random.rand(len(Y_iri)) < 0.5 
X_train_iri = X_iri[mask]
Y_train_iri = Y_iri[mask]
mask = np.logical_not(mask)
Y_test_iri = Y_iri[mask]
X_test_iri = X_iri[mask]
# 




## Linear regression : polynomial basis function
def Polybasis(X=None, L=None):
	basis = np.ones((len(X),L))
	for i in range(L):
		basis[:,i] = np.power(X,i+1)
	return basis
def Radialbasis(X=None,L=None):
	basis = np.zeros((len(X),L))
	mu = np.linspace(0, 60, num=L)
	sigma = 60/(L-1)
	for i in range(L):
		basis[:,i] = np.exp(-np.power(X-mu[i],2) / (2* sigma**2))
	return basis
def Logisticbasis(X=None):
	basis = np.hstack(np.ones((len(X),1)), X)
	return basis
def Linearfit(basis=None, Y_train=None):
	theta = np.linalg.solve(np.dot(basis.T,basis), np.dot(basis.T, Y_train))
	betainv = 1/len(Y_train) * (np.dot(Y_train.T, Y_train) - 2* np.dot(np.dot(theta.T, basis.T), Y_train) + np.dot(np.dot(basis, theta).T ,np.dot(basis, theta))) 
	return theta, betainv
def BayesianMAPfit(basis=None, Y_train=None,alpha=None, beta=None):
	I = np.identity(basis.shape[1])
	theta = np.linalg.solve(np.dot(basis.T,basis) + alpha/beta*I, np.dot(basis.T, Y_train))
	return theta
def Logisticfit(basis=None, Y_train=None):
	alpha = 0
	theta_init = np.ones(15)
	def f(x,basis,Y_train,alpha):
		Theta = np.zeros((5,3))
		for i in range(3):
			Theta[:,i] = x[i:i+5]
		return alpha / 2 * np.dot(x.T, x) - np.sum(np.dot(basis, Theta) * Y_train - np.log(np.sum(np.exp(np.dot(basis, Theta)), axis =1)))
	theta_hat = minimize(f,theta_init).x
	return theta_hat
def softmax(theta=None, X_test=None):
	Theta = np.zeros((5,3))
	for i in range(3):
		Theta[:,i] = x[i:i+5]
	x = np.dot(Logisticbasis(X_test), Theta).T
	e_x = np.exp(x)
	pred = e_x / np.sum(e_x,axis=0)
	return pred

def RMS(basis=None, theta=None, Y=None):
	RMS_error = np.sqrt(((np.dot(basis,theta)- Y) ** 2).mean())
	return RMS_error

def PolyLinear(X_train=None, Y_train=None, X_test=None, Y_test=None):
	L = np.linspace(1,20,num=20,dtype=np.int)
	RMS_train = np.zeros(len(L))
	RMS_test = np.zeros(len(L))
	theta = np.zeros((20, len(L)))
	betainv = np.zeros(len(L))
	for i in range(len(L)):
		basis = Polybasis(X_train,L[i])
		theta[0:L[i],i],betainv[i] = Linearfit(basis,Y_train)
		RMS_train[i] = RMS(basis, theta[0:L[i],i], Y_train)
		test_basis = Polybasis(X_test, L[i])
		RMS_test[i] = RMS(test_basis, theta[0:L[i],i], Y_test)

	L_opt_train = L[np.argmin(RMS_train)]
	theta_opt_train = theta[0:L_opt_train, np.argmin(RMS_train)]
	L_opt_test = L[np.argmin(RMS_test)]
	theta_opt_test = theta[0:L_opt_test, np.argmin(RMS_test)]
	# plot error
	plt.plot(L,RMS_train, linestyle='--', marker = '*', color = 'b', label = 'RMS error of training set')
	plt.plot(L,RMS_test, linestyle='--', marker = 'o', color = 'r', label = 'RMS error of testing set')
	plt.yscale("log")
	plt.legend()
	plt.show()
	# plot fit
	x =  np.linspace(0,50, num=200)
	y = np.dot(Polybasis(x,L_opt_train),theta_opt_train)
	plt.scatter(X_train, Y_train, marker='o', color = 'b', label = 'Training data')
	plt.plot(x,y, color = 'b', label = 'Best training fit')
	y = np.dot(Polybasis(x,L_opt_test),theta_opt_test)
	plt.scatter(X_test, Y_test, marker='*', color = 'r', label = 'Testing data')
	plt.plot(x,y, color = 'r', label = 'Best testing fit')
	plt.legend()
	plt.axis([0,60,-200,100])
	plt.show()

def RadLinear (X_train=None, Y_train=None, X_test=None, Y_test=None):
	L = np.array([5, 10, 15, 20, 25])
	RMS_train = np.zeros(len(L))
	RMS_test = np.zeros(len(L))
	theta = np.zeros((25, len(L)))
	betainv = np.zeros(len(L))
	for i in range(len(L)):
		basis = Radialbasis(X_train,L[i])
		theta[0:L[i],i],betainv[i] = Linearfit(basis,Y_train)
		RMS_train[i] = RMS(basis, theta[0:L[i],i], Y_train)
		test_basis = Radialbasis(X_test, L[i])
		RMS_test[i] = RMS(test_basis, theta[0:L[i],i], Y_test)

	L_opt_train = L[np.argmin(RMS_train)]
	theta_opt_train = theta[0:L_opt_train, np.argmin(RMS_train)]
	L_opt_test = L[np.argmin(RMS_test)]
	theta_opt_test = theta[0:L_opt_test, np.argmin(RMS_test)]
	# plot error
	plt.plot(L,RMS_train, linestyle='--', marker = '*', color = 'b', label = 'RMS error of training set')
	plt.plot(L,RMS_test, linestyle='--', marker = 'o', color = 'r', label = 'RMS error of testing set')
	plt.yscale("log")
	plt.legend()
	plt.show()
	# plot fit
	x =  np.linspace(0,50, num=200)
	y = np.dot(Radialbasis(x,L_opt_train),theta_opt_train)
	plt.scatter(X_train, Y_train, marker='o', color = 'b', label = 'Training data')
	plt.plot(x,y, color = 'b', label = 'Best training fit')
	y = np.dot(Radialbasis(x,L_opt_test),theta_opt_test)
	plt.scatter(X_test, Y_test, marker='*', color = 'r', label = 'Testing data')
	plt.plot(x,y, color = 'r', label = 'Best testing fit')
	plt.legend()
	plt.axis([0,60,-200,100])
	plt.show()

def BayesMAP(X_train=None, Y_train=None, X_test=None, Y_test=None):
	L = 50; beta = 0.0025
	alpha = np.logspace(-8,0,100)
	basis = Radialbasis(X_train, L)
	RMS_test = np.zeros(len(alpha))
	theta = np.zeros((L,len(alpha)))

	for i in range(len(alpha)):
		theta[:,i] = BayesianMAPfit(basis, Y_train, alpha[i], beta)
		test_basis = Radialbasis(X_test, L)
		RMS_test[i] = RMS(test_basis, theta[:,i], Y_test)
	alpha_opt = alpha[np.argmin(RMS_test)]
	theta_opt = theta[:,np.argmin(RMS_test)]
	
	plt.plot(alpha,RMS_test, linestyle='--', marker = 'o', color = 'r', label = 'RMS error of testing set')
	#plt.yscale("log")
	plt.legend()
	plt.show()
	#print("The optimal alpha is {}.".format(alpha_opt))
	x =  np.linspace(0,60, num=200)
	y = np.dot(Radialbasis(x,L),theta_opt)
	plt.scatter(X_test, Y_test, marker='*', color = 'r', label = 'Testing data')
	plt.plot(x,y, color = 'b', label = 'Best testing fit')
	plt.legend()
	plt.show()
	

def logisticReg(X_train=None, Y_train=None, X_test=None, Y_test=None):
	K = Y_train.shape[1]
	basis = Logisticbasis(X_train)
	theta_hat = Logisticfit(basis, Y_train)
	pred = softmax(theta_hat, X_test)
	pred_label = np.argmax(pred)

	# accuracy
	mask = pred_label == Y_iri_raw
	accuracy = np.sum(mask)/len(X_test)
	print('The accuracy of Logistic Regression is {}.'.format(accuracy))

def main():
	# load data set
	data = np.loadtxt("/Users/hurui/Downloads/CS 5783 ML/crash.txt")
	mask = np.arange(1,len(data)+1,2)
	data_test = data[::2]
	data_train = data[1::2]
	X_train = data_train[:,0]
	Y_train = data_train[:,1]
	X_test = data_test[:,0]
	Y_test = data_test[:,1]
	plt.scatter(X_train, Y_train)
	plt.show()

	#PolyLinear(X_train,Y_train,X_test,Y_test)
	#RadLinear(X_train,Y_train,X_test,Y_test)
	BayesMAP(X_train,Y_train,X_test,Y_test)
	#logisticReg(X_train_iri,Y_train_iri,X_test_iri,Y_test_iri)
if __name__ == "__main__":
	main()





