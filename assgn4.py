'''
Assignment 4 CS 5783 Machine Learning
@author: Rui Hu
'''
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
from scipy.stats import uniform as unif
import random
import pickle
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns


###################################### Gaussian Process Regression

data = np.loadtxt("Downloads/crash.txt")
X = data[:,0]/np.amax(data[:,0])
Y = data[:,1]/np.amax(data[:,1])
def kernel_squared_exponential(x, y, delta):
	dx = np.expand_dims(x, 1) - np.expand_dims(y, 0)
	return np.exp(-1/2.0 * np.power(dx, 2) / delta**2)
def kernel_exponential(x,y,delta):
	dx = np.expand_dims(x, 1) - np.expand_dims(y, 0)
	return np.exp(-1/2.0 * np.abs(dx) / delta)
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

# choose 0.01-1
Delta = np.linspace(0.01,1, num=100)
# 5-fold
kf = cross_validation.KFold(len(X), n_folds=5)
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

print("The optimal delta for squared exponential kernel is {}.".format(opt_delta))

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

##################################### K-means #########################
## K-means clustering
# dataset

with open ("C:/Users/Administrator/Downloads/t10k-images.idx3-ubyte", 'rb') as f:
	X_raw = f.read()
X_byte = bytearray(X_raw) 
X = np.asarray(X_byte[16:]).reshape([10000,28*28])
with open ("C:/Users/Administrator/Downloads/t10k-labels.idx1-ubyte", 'rb') as f:
	testing_labels_raw = f.read()
testing_labels_byte = bytearray(testing_labels_raw) 
testing_labels = np.asarray(testing_labels_byte[8:]).reshape([10000,])


def centroid_fit(points, centroids):
	"""returns an array containing the index to the nearest centroid for each point"""
	distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
	return np.argmin(distances, axis=0), np.amin(distances, axis=0)
def centroid_update(points, closest, centroids):
	"""returns the new centroids assigned from the points closest to them"""
	return np.array([points[closest==k].mean(axis=0) for k in range(centroids.shape[0])])
# initialization 
def init(X,K,init_type):
	if init_type == "1":
		idx = np.random.randint(len(X), size=K)
		init_u = X[idx]
	elif init_type =="2":
		init_u = X[np.random.randint(len(X), size=1)]
		while len(init_u) < K:
			closest, dist = centroid_fit(X,init_u)
			init_u.append(centroid_update(X,closest,init_u))
	elif init_type == "3":
		init_u = np.zeros((K,X.shape[1]))
		for i in range(K):
			mask = testing_labels == i
			init_u[i] = X[mask[0]]
		
	return init_u
def assign(K,X,init_type):
	c = init(X, K, init_type)
	while True:
		old_c = np.copy(c)
		closest, min_dist = centroid_fit(X, c)
		obj = np.sum(min_dist)
		print(obj)
		c = centroid_update(X, closest, c)
		if np.all(c == old_c):
			break
	# one-hot labels
	Y = np.array(closest, dtype = np.int)
	assignment = np.eye(K)[Y]
	# centroid
	Final_centroid = c
	return assignment, Final_centroid
K=10
assignment_1, centroid_1 = assign(K,X,init_type="1")
assignment_2, centroid_2 = assign(K,X,init_type="2")
assignment_3, centroid_3 = assign(K,X,init_type="3")

# 
K=3
assignment, centroid = assign(K,X,init_type="3")
# show the images in each class
for i in range(len(centroid)):
	img = centroid[i]
	plt.imshow(img.reshape((28,28)))
	plt.title('Cluster mean images')
	plt.show()
	img_c = X[assignment == i][0:3]
	plt.imshow(img.reshape((28,28)))
	plt.show()



######################################## HMM ##########################
# dishonest casino
# Number of hidden states:
# State = (['F','L'])
# Number of distinct observation symbols:
# V = ([1,2,3,4,5,6])
# state transition probability distribution:
# A = [[0.95 0.05],
#      [0.01 0.90]]
# observation symbol probability distribution :
# B = [[0.1666 0.1666 0.1666 0.1666 0.1666 0.1666],
#      [0.1 0.1 0.1 0.1 0.1 0.5]]
# initial state distribution:
# pi = [0.5 0.5]
#

# Observation sequence generation:
random.seed(42)
state = ['Fair', 'Loaded']
V = [1, 2, 3, 4, 5, 6]
B = [[1/6, 1/6, 1/6, 1/6, 1/6, 1/6],[1/10, 1/10, 1/10, 1/10, 1/10, 5/10]]

A = [[0.95, 0.05],[0.1, 0.90]]

pi = [0.5, 0.5]

def rolling_dice(T):
	die = [ ]
	rolls = [ ]
	# Decide starting dice
	start_rv = unif.rvs()
	if start_rv <= pi[0]:
		dice = 0
	else:
		dice = 1
	die.append(state[dice])
	# Roll the dice for n times
	for i in range(T):
		face = random.choices(V, B[dice])[0]
		rolls.append(face)
		trans_rv = unif.rvs()
		if trans_rv <= A[dice][dice]:
			dice = dice
		else:
			dice ^= 1
		if i < T - 1:
			die.append(state[dice])
	return rolls, die


T = 1000
rolls, die = rolling_dice(T)
enc = OneHotEncoder()
rolls_one_hot = enc.fit_transform(rolls.values.reshape(-1, 1)) # one-hot

class HMM:
	def __init__(self, pi=pi, A=A, B=B, roll=rolls_one_hot):
		self.pi = pi
		self.A = A
		self.B = B
		self.roll = roll
		self.N = len(self.pi) # num of hidden state
		self.T, self.M = self.roll.shape # num of trails, num of Number of distinct observation symbols
		self.Z = np.zeros(self.T) # initiate prob of observing O_t given O_t-1 at time t
		self.alpha = np.zeros((self.T, self.N)) # initiate prob of hidden state N=0 or 1 given O_0:t and system parameter at time t

	def normalize(self, a):
		return a / np.sum(a)

	def find_B_t_O(self, t): # find b_i(O_t) to compute alpha(t,i) = pi(i) * b_i(O_t) or alpha(t+1, j) = sum_i(alpha(t,i)) * b_j(O_t+1) 
		result = []
		for i in range(self.N): # for two hidden state
			result.append(list(compress(self.B[i], self.roll[t]))[0])  # b_i(O_t)
			#compress(condition, a, axis=None, out=None)[source];Return selected slices of an array along given axis.
		return result

	def forwards(self):
		# p(q(t)=S_i|O(1), ...., O(t))
		# when t = 0 (first)
		alpha_u = np.multiply(self.pi, self.find_B_t_O(0)) # alpha(t,i) = pi(i) * b_i(O_t) when t=0
		self.alpha[0] = self.normalize(alpha_u)
		self.Z[0] = np.sum(alpha_u) # Z(t) = p(O(t)|O(1), ...., O(t-1)) = sum_i(alpha(t,i) = pi(i) * b_i(O_t))
		logZ = np.log(self.Z[0] + 1e-7)
		# when t>0，alpha(t+1, j) = sum_i(alpha(t,i)) * b_j(O_t+1) 
		for t in range(1, self.T):
			alpha_u = np.multiply(np.dot(self.alpha[t-1], self.A), self.find_B_t_O(t))
			self.alpha[t] = self.normalize(alpha_u)
			self.Z[t] = np.sum(alpha_u)
			logZ += np.log(self.Z[t] + 1e-7) # sum_t sum_i (alpha(T,i))
		return self.alpha, self.Z, logZ
		
		def backwards(self):
			self.beta[-1] = np.ones(self.N) # beta_t(i) = prob( O_t+1:T | q(t)=S_i)
			for t in range(self.T - 2, -1, -1): # backward
				self.beta[t] = self.normalize(np.dot(np.transpose(self.A),
					np.multiply(self.find_B_t_O(t + 1), self.beta[t + 1])))
			return self.beta
		def backward_smooth(self):
			# gamma(t, i) = p(S(t) = i|O(1:T)) = alpha(t,i) * beta(t,i)
			self.alpha, _, logZ = self.forwards()
			self.beta = self.backwards()
			gamma = np.multiply(self.alpha, self.beta)
			# plot gamma
			# normalize gamma
			sum_ = gamma.sum(1)
			for n in range(self.N):
				gamma[:, n] = gamma[:, n]/ sum_
			plot(gamma)
			return gamma
		## MAP estimation: argmax prob (q(0:T)| O_1:T)
		def viterbi(self):
			# Initialization
			delta = np.zeros((self.T, self.N)) 
			# maximum probability of a single path that ends in the state i at time t, accounting for first t observations.
			fai = np.zeros((self.T, self.N), dtype=np.int)
			optimal_path = np.zeros(self.T, dtype=np.int)
			# when t=0
			delta[0] = np.multiply(self.pi, self.find_B_t_O(0))
			# fai[0] = 0
			# recursion
			for t in range(1, self.T):
				for n in range(self.N):
					delta[t, n] = np.max(delta[t-1] * self.A[:, n]) * self.find_B_t_O(t)[n]
					fai[t, n] = np.argmax(delta[t-1] * self.A[:, n]) 
					# Termination
					optimal_p = np.max(self.delta[-1])
					optimal_path[-1] = np.argmax(delta[-1])
					# Backtracking the path
					for t in range(self.T-2, -1, -1):
						optimal_path[t] = fai[t + 1, optimal_path[t + 1]]
			return p_star, optimal_path


hmm = HMM()
alpha, _, _ = hmm.get_belief_state()

gamma = hmm.backward_smooth()
p_star, q_t = hmm.viterbi()


# 
# Encoding die as categorical variable
die_ = pd.DataFrame(die, columns=['die'])
die_.die = pd.Categorical(die_.die)
die_['code'] = die_.die.cat.codes

# plot the estimated alpha and actual hidden states
color_ = sns.color_palette("hls", 8)[5]
plt.fill_between(range(T), 0, die_['code'],
				 color=color_, alpha=0.2)
alpha_ = pd.DataFrame(alpha)
alpha_.iloc[:, 1].plot(color=color_, linewidth=2)
plt.xlabel("# trials")
plt.ylabel("probability of loaded dice")
plt.title('Output of forwards')
plt.show()

#
#alpha_ = pd.DataFrame(gamma)
#alpha_.iloc[:, 1].plot(color=color_, linewidth=2)
#plt.xlabel("# trials")
#plt.ylabel("probability of loaded dice")
#plt.show()
# 
plt.fill_between(range(T), 0, die_['code'],
				 color=color_, alpha=0.2)
alpha_ = pd.DataFrame(q_t)
alpha_.iloc[:, 1].plot(color=color_, linewidth=2)
plt.xlabel("# trials")
plt.ylabel("probability of loaded dice")
plt.title("Output of the MAP estimation")
plt.show()
