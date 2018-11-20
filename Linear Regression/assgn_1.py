import numpy as np
import matplotlib.pyplot as plt
'''
Assignment 1
@author: Rui Hu
'''

class KDTree:
	'''
	KDTree:
	dimension: dimension of data
	right: sub-tree on the right 
	left: sub-tree on the left 
	axis: current dimension to be cut
	'''
	def __init__(self, data=None):
		'''
		split the data along the median of 
		each feature vector in turn
		'''
		# define an inner class Node
		class Node:
			'''
			Node:
			data_point: data points
			split: the dimension to be cut
			left: sub-tree on the left 
			right: sub-tree on the right
			'''
			def __init__(self, point=None, split=None, left=None, right=None):
				self.point = point
				self.split = split
				self.left = left
				self.right = right
		def createNode(split=None, data=None):
			if  len(data) == 0:
				return None
			# determine the dimension to be cut
			dimension = len(data[0]) # all have the same dimension
			axis = split % dimension
			# sort all the data points
			data=list(data)
			data.sort(key=lambda x: x[axis])
			data = np.array(data)
			median = len(data) // 2
			return Node(data[median], axis, createNode(axis+1,data[:median]), createNode(axis+1,data[median+1:])) 

		# find the root of the tree: first node of cutting all data into two parts
		self.root = createNode(0,data)

	def find_nearest(self, target_point):
		'''
		Input: the data point to be classified
		Output: the nearest neighbor of the Input
		'''
		def distance(node_1, node_2):
			''' 
			Euclidean Distance between two points
			'''
			dist = np.sqrt(np.sum(np.square(node_1 - node_2)))
			return dist

		temp_node = self.root
		NN = [temp_node.point]
		min_dist_array = [float("inf")]
		node_list = []
		def SearchPath(temp_node=None, node_list=None, min_dist_array=None, NN=None, target_point=None):
			while temp_node :
				node_list.append(temp_node)
				split = temp_node.split
				point = temp_node.point
				tmp_dist = distance(point,target_point)
				if tmp_dist < np.max(min_dist_array) :   
					min_dist_array = tmp_dist
					NN = point
				if target_point[split] <= point[split] :
					temp_node = temp_node.left
				else :
					temp_node = temp_node.right
			return NN,min_dist_array
		NN,min_dist_array = SearchPath(temp_node,node_list,min_dist_array, NN, target_point)
	
		while node_list :
			back_node = node_list.pop()
			split = back_node.split
			point = back_node.point
			if not abs(target_point[split] - point[split]) >= np.max(min_dist_array) :
				if (target_point[split] <= point[split]) :
					temp_node = back_node.right
				else : 
					temp_node = back_node.left

				if temp_node :
					NN,min_dist_array = SearchPath(temp_node,node_list,min_dist_array, NN, target_point)
		return NN, min_dist_array

def generateData(num_dis=None):
	X =[];Y=[]
	if num_dis == 2:
		mean_1 = [0,2]
		cov_1 = [[1,.75],[.75,1]]
		mean_2 = [3.5,4]
		cov_2 = [[1,.5],[.5,1]]
		size = 5000
		X1 = np.random.multivariate_normal(mean_1, cov_1, size)
		X2 = np.random.multivariate_normal(mean_2, cov_2, size)
		X = np.vstack([X1, X2])
		Y = np.hstack([np.zeros(size),np.ones(size)])
	elif num_dis == 10:
		X = np.zeros((10000,2))
		Y = np.zeros(10000)
		Size = np.array([1000,1000,1000,1000,1000])
		MEAN = np.array([[2.5,2],[1,4],[4.5,3],[2.5,-1],[0,2],[0,0],[0,4],[4,0],[4,6],[6,4]])
		COV = np.array([0.75,0,0.6,0.8,0,0.7,0.6,0,0.75,0.8])*1.2
		for i in range(5):
			mean_1 =MEAN[i,:];
			mean_2=MEAN[i+5,:];
			cov_1=[[.8,COV[i]],[COV[i],.8]];
			cov_2=[[.7,COV[i+5]],[COV[i+5],.7]];
			X1 = np.random.multivariate_normal(mean_1, cov_1, Size[i])
			X2 = np.random.multivariate_normal(mean_2, cov_2, Size[i])
			if i ==0:
				X[0:Size[i]*2,:] = np.vstack([X1, X2])
				Y[0:Size[i]*2] = np.hstack([np.zeros(Size[i]),np.ones(Size[i])])
			else:
				X[np.sum(Size[0:i])*2:np.sum(Size[:i])*2+Size[i]*2,:] = np.vstack([X1, X2])
				Y[np.sum(Size[0:i])*2:np.sum(Size[:i])*2+Size[i]*2]= np.hstack([np.zeros(Size[i]),np.ones(Size[i])])

	plt.scatter(X[:,0], X[:,1], c = Y, marker='x', alpha=0.5)
	plt.show()

	# partition: traning set (75%) and test set (25%)
	mask = np.random.rand(len(X)) < 0.75 # keep the random number the same
	X_train = X[mask,:]
	Y_train = Y[mask]
	mask = np.logical_not(mask)
	X_test = X[mask,:]
	Y_test = Y[mask]
	return X_train, Y_train, X_test, Y_test

class LinearClassifier():
	'''
	Linear Classification
	'''
	def __init__(self):
		self.beta = None

	def fit(self, X=None, Y=None):
		# add a dimension on X
		X = np.insert(X, 0, 1, axis=1)
		# X = X[:, np.newaxis]
		self.beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

	def predict(self, X):
		'''
		output the lable y of test data X
		'''
		# add a dimension on X
		X = np.insert(X, 0, 1, axis=1)
		#X = X[:, np.newaxis]
		# predict
		pred = X.dot(self.beta)
		# initialize Y
		Y_pred = np.ones(len(X))
		# lable = 0 if pred < 0.5; o.w. label = 1
		mask = pred < 0.5
		Y_pred[mask] = 0
		return Y_pred

def accuracyClassification(y_pred=None, y_test=None):
	mask = y_pred == y_test
	accuracy = np.sum(mask)/len(y_test)
	return accuracy

def LinearClassification(X_train=None, Y_train=None, X_test=None, Y_test=None):
	'''
	classify the test sample using linear classification
	'''
	# train a linear classification model
	model = LinearClassifier()
	model.fit(X_train,Y_train)

	# predict on the test set
	Y_pred = model.predict(X_test)
	accuracy = accuracyClassification(Y_pred,Y_test)
	print('Linear_accuracy =', accuracy)
	# display:
	# Training set elements from class 0
	X_train = np.array(X_train)
	mask = Y_train == 0
	plt.scatter(X_train[mask,0], X_train[mask,1],label = 'training_set_0', color = 'b', marker = 'o', alpha = 0.4)
	# Training set elements from class 1
	mask = np.logical_not(mask)
	plt.scatter(X_train[mask,0], X_train[mask,1],label = 'training_set_1', color = 'y', marker = 'x', alpha = 0.4)
	
	# Correctly classified test set elements from class 0
	mask = Y_test == 0
	X_Test = X_test[mask]
	mask= Y_pred[mask] == Y_test[mask]

	plt.scatter(X_Test[mask,0], X_Test[mask,1],label = 'test_set_cor_0', color = 'g', marker = 'o', alpha = 0.6)
	# Incorrectly classified test set elements from class 0
	mask = np.logical_not(mask)
	plt.scatter(X_Test[mask,0], X_Test[mask,1],label = 'test_set_incor_0', color = 'r', marker = 'o',alpha=0.6)
	
	# Correctly classified test set elements from class 1
	mask = Y_test == 1
	X_Test = X_test[mask]
	mask = Y_pred[mask] == Y_test[mask]
	plt.scatter(X_Test[mask,0], X_Test[mask,1],label = 'test_set_cor_1', color = 'g', marker = 'x', alpha = 0.6)
	# incorrectly classified test set elements from class 1
	mask = np.logical_not(mask)
	plt.scatter(X_Test[mask,0],X_Test[mask,1], label = 'test_set_incor_1', color = 'r', marker = 'x', alpha = 0.6)
	plt.legend()
	plt.show()
	
	return accuracy

def KnnClassification(X_train=None, Y_train=None, X_test=None, Y_test=None):
	''' 
	classify test samples using KNN
	'''
	# Find the 1-nearest neighbor
	KD = KDTree(X_train)
	X_train = X_train.tolist()
	Y_pred = np.zeros(len(X_test))
	for i in range(len(X_test)):
		NN, dis_min = KD.find_nearest(X_test[i])
		nni = list(NN)
		Y_pred[i] = Y_train[X_train.index(nni)]
	X_train = np.array(X_train)
	accuracy = accuracyClassification(Y_pred,Y_test)
	print('KNN_accuracy = ', accuracy)
	# display:
	# Training set elements from class 0
	mask = Y_train == 0
	plt.scatter(X_train[mask,0], X_train[mask,1],label = 'training_set_0', color = 'b', marker = 'o', alpha = 0.4)
	# Training set elements from class 1
	mask = np.logical_not(mask)
	plt.scatter(X_train[mask,0], X_train[mask,1],label = 'training_set_1', color = 'y', marker = 'x', alpha = 0.4)
	
	# Correctly classified test set elements from class 0
	mask = Y_test == 0
	X_Test = X_test[mask]
	mask= Y_pred[mask] == Y_test[mask]

	plt.scatter(X_Test[mask,0], X_Test[mask,1],label = 'test_set_cor_0', color = 'g', marker = 'o', alpha = 0.6)
	# Incorrectly classified test set elements from class 0
	mask = np.logical_not(mask)
	plt.scatter(X_Test[mask,0], X_Test[mask,1],label = 'test_set_incor_0', color = 'r', marker = 'o',alpha=0.6)
	
	# Correctly classified test set elements from class 1
	mask = Y_test == 1
	X_Test = X_test[mask]
	mask = Y_pred[mask] == Y_test[mask]
	plt.scatter(X_Test[mask,0], X_Test[mask,1],label = 'test_set_cor_1', color = 'g', marker = 'x', alpha = 0.6)
	# incorrectly classified test set elements from class 1
	mask = np.logical_not(mask)
	plt.scatter(X_Test[mask,0],X_Test[mask,1], label = 'test_set_incor_1', color = 'r', marker = 'x', alpha = 0.6)
	plt.legend()
	plt.show()
	
	return accuracy
def main():
	## 1st case: 2 distribution
	# generate random data
	X_train, Y_train, X_test, Y_test = generateData(num_dis=2)

	# linear classification
	Linear_accuracy = LinearClassification(X_train, Y_train, X_test, Y_test)

	# KNN classification
	KNN_accuracy= KnnClassification(X_train, Y_train, X_test, Y_test)

	
	## 2nd case: 10 distribution
	X_train, Y_train, X_test, Y_test = generateData(num_dis=10)
	# linear classification
	Linear_accuracy_10 = LinearClassification(X_train, Y_train, X_test, Y_test)

	# KNN classification
	KNN_accuracy_10= KnnClassification(X_train, Y_train, X_test, Y_test)


if __name__ == "__main__":
	main()