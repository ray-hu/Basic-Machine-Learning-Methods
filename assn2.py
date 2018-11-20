'''
Assignment 2 CS 5783 Machine Learning
@author: Rui Hu
'''
#! wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
# ！gzip -d train-images-idx3-ubyte.gz
from __future__ import division
import numpy as np
from scipy.spatial import cKDTree 
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from scipy.stats import mode
import warnings


def dataExtract():

	with open ("/Users/hurui/Downloads/CS 5783 ML/train-images-idx3-ubyte", 'rb') as f:
		training_images_raw = f.read()

	training_images_byte = bytearray(training_images_raw) 
	training_images = np.asarray(training_images_byte[16:]).reshape([60000,28*28])

	with open ("/Users/hurui/Downloads/CS 5783 ML/train-labels-idx1-ubyte", 'rb') as f:
		training_labels_raw = f.read()

	training_labels_byte = bytearray(training_labels_raw) 
	training_labels = np.asarray(training_labels_byte[8:]).reshape([60000,])

	with open ("/Users/hurui/Downloads/CS 5783 ML/t10k-images-idx3-ubyte", 'rb') as f:
		testing_images_raw = f.read()

	testing_images_byte = bytearray(testing_images_raw) 
	testing_images = np.asarray(testing_images_byte[16:]).reshape([10000,28*28])

	with open ("/Users/hurui/Downloads/CS 5783 ML/t10k-labels-idx1-ubyte", 'rb') as f:
		testing_labels_raw = f.read()

	testing_labels_byte = bytearray(testing_labels_raw) 
	testing_labels = np.asarray(testing_labels_byte[8:]).reshape([10000,])
	#print( training_labels.shape, training_images[0].shape)
	return training_images,training_labels,testing_images,testing_labels
def acc(pred_labels, testing_labels):
	# accuracy: test error rate
	accuracy  = np.sum(pred_labels == testing_labels)/ len(testing_labels)
	return accuracy
def NBclassification(training_images = None, training_labels=None, testing_images=None, testing_labels=None):
	# Naive Bayes classification
	def dataGenerate(training_images=None, testing_images=None):
		
		mask = training_images > 100
		training_images[mask] = 1
		training_images[np.logical_not(mask)]=0
		mask = testing_images > 100
		testing_images[mask] = 1
		testing_images[np.logical_not(mask)]=0
		return training_images, testing_images
	def naiveBayes(training_images = None, training_labels=None):
		# Naive Bayes Classifier
		# prior distribution of theta (uniform Dirichlet)
		# Uniform Dir = Uniform --> each category has the same probability
		category_num = np.ones(10,dtype=int)
		dim_num = np.ones((10,28*28),dtype = int)
		theta = np.zeros((10,28*28), dtype = float)
		for i in range(10):
			mask = training_labels == i
			category_num[i] = category_num[i] + sum(mask)
			dim_num[i,:] = dim_num[i,:] + np.sum(training_images[mask,:], axis=0) 
			theta[i,:] = theta[i,:] + dim_num[i,:] / category_num[i]
		return theta
	def posterior(image=None):
		# posterior
		tmp = np.zeros(10)
		for c in range(10):
			tmp[c] = tmp[c] + theta_prior[c]
			mask = (image == 1)
			tmp[c] = tmp[c] + sum(theta_log[c,mask]) + sum(theta_log_0[c,np.logical_not(mask)])
		pred_labels = np.argmax(tmp)
		return pred_labels

	# process data into binary
	training_images, testing_images = dataGenerate(training_images, testing_images)
	# log_theta
	theta = naiveBayes(training_images, training_labels)
	theta_log = np.log(theta)
	theta_log_0 = np.log(1-theta)
	# show the theta
	#plt.matshow(theta_log[3,:].reshape(28,28))
	#plt.colorbar()
	#plt.show()

	# prior
	theta_prior = np.log(np.ones(10)*0.1) # uniform

	# prediction
	pred_labels = np.zeros(len(testing_images))
	for j in range(len(testing_images)):
		pred_labels[j] = posterior(testing_images[j])
	accuracy = acc(pred_labels, testing_labels)
	print('The accuracy of NB is {}.'.format(accuracy))
	return pred_labels, accuracy
def NbGaussianClassification(training_images = None, training_labels=None):
	# chech random sample
	# process data
	def dataGenerate(training_images = None, training_labels=None):
		np.random.seed(0)
		idx = np.random.shuffle(np.arange(len(training_images)))
		training_images = training_images[idx]
		training_labels = training_labels[idx]

		images = training_images[training_labels == 5][:1000]
		labels = training_labels[training_labels == 5][:1000]
		labels[:] = 1
		#print(images.shape)
		images = np.vstack((images, training_images[training_labels != 5][:1000]))
		tmp_labels = training_labels[training_labels != 5][:1000]
		tmp_labels[:] = 0
		labels = np.hstack((labels, tmp_labels))
		# training and testing sets
		#print(images.shape, labels.shape)
		np.random.seed(5)
		mask = np.random.rand(len(images)) < 0.9
		training_images = np.asarray(images[mask])
		training_labels = np.asarray(labels[mask])
		testing_images = np.asarray(images[np.logical_not(mask)])
		testing_labels = np.asarray(labels[np.logical_not(mask)])
		return training_images, training_labels, testing_images, testing_labels

	def parameterFit(training_images=None, training_labels=None):
		mu = np.zeros((2,28*28), dtype = float)
		sigma = np.zeros(10, dtype = float)
		for i in range(2):
			mask =  training_labels==i
			mu[i,:] = np.mean(training_images[mask,:], axis=0)
			sigma[i] = np.var(training_images[mask,:])

		return mu,sigma

	def predict(image=None, mu=None, sigma=None, lambda_ratio=None, q=None):
		tao = np.log(q / (1-q) * lambda_ratio)
		tmp = np.sum(-(image-mu[1,:])**2 / (2*sigma[1]) + (image-mu[0,:])**2 / (2*sigma[0]))
		if tmp > tao:
			pred_label = 1
		else:
			pred_label = 0
		return pred_label


	# data
	training_images, training_labels, testing_images, testing_labels = dataGenerate(training_images, training_labels)
	# lambda ratio 10/01
	lambda_ratio = np.asarray([5,2,1,0.5,0.2])
	# fit parameters
	mu,sigma = parameterFit(training_images, training_labels)
	# prediction
	pred_labels = np.zeros((5, len(testing_images)))
	mask = testing_labels == 1
	TPR = np.zeros(len(lambda_ratio))
	FPR = np.zeros(len(lambda_ratio))
	for r in range(len(lambda_ratio)):
		for i in range(len(testing_images)):
			pred_labels[r,i] = predict(testing_images[i], mu, sigma, lambda_ratio[r], 0.5)
		# TPR FPR
		TP = sum(pred_labels[r,:][mask] == 1)
		FN = sum(pred_labels[r,:][mask] == 0)
		FP = sum(pred_labels[r,:][np.logical_not(mask)] == 1)
		TN = sum(pred_labels[r,:][np.logical_not(mask)] == 0)
		TPR[r] = TP /(TP + FN)
		FPR[r] = FP / (FP + TN)
		#accuracy = acc(pred_labels[r,:], testing_labels)
		#print(accuracy)
	# ROC
	plt.plot(FPR, TPR, linestyle='--', marker='o', color='b')
	plt.xlabel('False Positive Rate (FPR)')
	plt.ylabel('True Positive Rate (TPR)')
	plt.title('ROC')
	#plt.axis([0,1,0,1])
	plt.show()
	# TPR FPR
def KNNclassification(training_images=None, training_labels=None, testing_images=None, testing_labels=None):
	
	def dataGenerate(training_images=None, training_labels=None, testing_images=None, testing_labels=None):
		# shuffle the data
		np.random.seed(0)
		idx = np.random.shuffle(np.arange(len(training_images)))
		training_images = training_images[idx]
		training_labels = training_labels[idx]
		idx = np.random.shuffle(np.arange(len(testing_images)))
		testing_labels = testing_labels[idx]
		testing_images = testing_images[idx]
		# 
		images = training_images[training_labels == 0][:200]
		labels = training_labels[training_labels==0][:200]
		images = np.vstack((np.vstack((images, training_images[training_labels == 1][:200])), training_images[training_labels == 6][:200]))
		labels = np.hstack((np.hstack((labels, training_labels[training_labels == 1][:200])), training_labels[training_labels == 6][:200]))
		training_images = np.asarray(images)
		training_labels = np.asarray(labels)
		images = testing_images[testing_labels == 0][:50]
		labels = testing_labels[testing_labels ==0][:50]
		images = np.vstack((np.vstack((images, testing_images[testing_labels  == 1][:50])), testing_images[testing_labels == 6][:50]))
		labels = np.hstack((np.hstack((labels, testing_labels[testing_labels == 1][:50])), testing_labels[testing_labels == 6][:50]))
		testing_images = np.asarray(images)
		testing_labels = np.asarray(labels)
		return training_images, training_labels, testing_images, testing_labels
	def KnnClassifier(training_images=None, training_labels=None, testing_images=None, testing_labels=None, k=None):
		# use scipy.spatial.cKDTree
		tree = cKDTree(training_images)
		#print(tree.data[0])
		[d,i] = tree.query(testing_images, k=k)
		pred_labels, count= mode(training_labels[i].T)
		accuracy = acc(pred_labels, testing_labels)
		#print(accuracy)
		return pred_labels,accuracy
	def cross_val(training_images=None, training_labels=None, k=None, fold=None):
		# split the data
		kf = KFold(n_splits=fold)

		#kf = KFold(training_images.shape[0], n_folds = fold, shuffle=True)
		accuracy = 0.0
		for train_idx, test_idx in kf.split(training_images):
			train_x, test_x = training_images[train_idx], training_images[test_idx]
			train_y, test_y = training_labels[train_idx], training_labels[test_idx]
			#print(train_x.shape, test_y.shape)
			pred, accu = KnnClassifier(train_x, train_y, test_x, test_y,k)
			accuracy = accuracy + accu
			#print(accuracy)
		accuracy = accuracy / fold
		return accuracy
		#print(accuracy)
	
	#data
	training_images, training_labels, testing_images, testing_labels = dataGenerate(training_images, training_labels, testing_images, testing_labels)
	#print(testing_images.shape)
	# cross validation
	k_range = np.asarray([1,3,5,7,9])
	cv_fold = 5
	accuracy = np.zeros(len(k_range))
	for i in range(len(k_range)):
		accuracy[i] = cross_val(training_images, training_labels, k_range[i], fold = cv_fold)
		#print(accuracy[i])

	best_k = k_range[np.argmax(accuracy)]
	# prediction
	pred_labels, acc_best = KnnClassifier(training_images, training_labels, testing_images, testing_labels,best_k)
	#print(best_k, acc_best)

	# result analysis
	pred_labels = np.asarray(pred_labels).reshape(testing_labels.shape)
	mask = pred_labels != testing_labels
	img = testing_images[mask,:]
	img_label = pred_labels[mask]

	print('The best k is {}.'.format(best_k))
	print('The accuracy of KNN is {}.'.format(acc_best))
	for i in range(len(img)):
		plt.imshow(img[i].reshape((28,28)))
		plt.title('Incorrectly Classified')
		plt.show()
		print('Incorrectly Classified as {}.'.format(img_label[i]))
	img = testing_images[np.logical_not(mask),:]
	plt.imshow(img[3].reshape((28,28)))
	plt.title('Correctly Classified')
	plt.show()
	return best_k, acc_best
def main():
	# Naive Bayes
	training_images,training_labels,testing_images,testing_labels = dataExtract()
	#print(testing_labels[9995:])
	plt.imshow(testing_images[1,...].reshape((28,28)))
	plt.show()
	pred_labels_NB, accuracy_NB = NBclassification(training_images,training_labels,testing_images,testing_labels)
	#print(accuracy_NB)

	# Naive Bayes Gaussian
	NbGaussianClassification(training_images, training_labels)

	# KNN
	best_k, acc_best = KNNclassification(training_images, training_labels, testing_images, testing_labels) 


if __name__ == "__main__":
	main()

