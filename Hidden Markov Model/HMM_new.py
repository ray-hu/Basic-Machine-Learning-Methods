#!/usr/bin/env python
# coding: utf-8

# In[264]:


import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import uniform as unif
import random
import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns


# In[265]:


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
state = ['F', 'L']
V = [1, 2, 3, 4, 5, 6]
B = [[1/6, 1/6, 1/6, 1/6, 1/6, 1/6],[1/10, 1/10, 1/10, 1/10, 1/10, 5/10]]
A = np.array([[0.95, 0.05],[0.1, 0.90]])
pi = [0.5, 0.5]


# In[266]:


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


# In[267]:


T = 1000
rolls, die = rolling_dice(T)
enc = OneHotEncoder()
rolls_one_hot = enc.fit_transform(np.reshape(rolls,(-1, 1))) # one-hot


# In[268]:


class HMM:
    def __init__(self, pi=pi, A=A, B=B, roll=rolls_one_hot):
        self.pi = pi
        self.A = A
        self.B = B
        self.roll = roll
        self.N = len(self.pi) # num of hidden state
        self.T, self.M = self.roll.shape # num of trails, num of distinct observation symbols
        self.Z = np.zeros(self.T) # initiate prob of observing O_t given O_t-1 at time t
        self.alpha = np.zeros((self.T, self.N)) # initiate prob of hidden state N=0 or 1 given O_0:t and system parameter at time t
        self.beta = np.zeros((self.T, self.N))
        
        
    def normalize(self, a):
        return a / np.sum(a)
    
    def find_B_t_O(self,t): # find b_i(O_t) to compute alpha(t,i) = pi(i) * b_i(O_t) or alpha(t+1, j) = sum_i(alpha(t,i)) * b_j(O_t+1) 
        result = []
        for i in range(self.N):
            result.append(self.B[i][np.argmax(self.roll[t])])
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
        for t in range(self.T -2, -1, -1): # backward
            self.beta[t] = self.normalize(np.dot(np.transpose(self.A),
                                                np.multiply(self.find_B_t_O(t + 1), self.beta[t+1])))
        return self.beta
    
    def backward_smooth(self):
        self.alpha, _, logZ = self.forwards()
        self.beta = self.backwards()
        gamma = np.multiply(self.alpha, self.beta)
        # normalize gamma
        sum_ = gamma.sum(1)
        for n in range(self.N):
            gamma[:,n] = gamma[:,n]/ sum_
        return gamma
    
    
    ## MAP estimation: argmax prob (q(0:T)| O_1:T)
    def viterbi(self):
        # Initialization
        delta = np.zeros((self.T, self.N))
        # 
        fai = np.zeros((self.T, self.N), dtype = np.int)
        optimal_path = np.zeros(self.T, dtype = np.int)
        # when t = 0
        delta[0] = np.multiply(self.pi, self.find_B_t_O(0))
        # fai[0] = 0
        # recursion
        for t in range(1, self.T):
            for n in range(self.N):
                delta[t, n] = np.max(delta[t-1] * self.A[:,n]) * self.find_B_t_O(t)[n]
                fai[t, n] = np.argmax(delta[t-1] * self.A[:, n])
        # Termination
        optimal_p = np.max(delta[-1])
        optimal_path[-1] = np.argmax(delta[-1])
        # backtracking the path
        for t in range(self.T - 2, -1, -1):
            optimal_path[t] = fai[t + 1, optimal_path[t + 1]]
        return optimal_p, optimal_path


# In[269]:


hmm = HMM()
alpha, _, _ = hmm.forwards()
p_star, q_t = hmm.viterbi()


# In[270]:


# Encoding die as categorical variable
die_ = pd.DataFrame(die, columns=['die'])
die_.die = pd.Categorical(die_.die)
die_['code'] = die_.die.cat.codes


# In[271]:


# plot the estimated alpha and actual hidden states
color_ = sns.color_palette("hls", 8)[6]
plt.fill_between(range(T), 0, die_['code'], color=color_, alpha=0.2)
alpha_ = pd.DataFrame(alpha)
alpha_.iloc[:, 1].plot(color=color_, linewidth=2)
plt.xlabel("# trials")
plt.ylabel("probability of loaded dice")
plt.title('Output of forwards')
plt.show()


# In[272]:


gamma = hmm.backward_smooth()


# In[273]:


# plot the estimated alpha and actual hidden states
color_ = sns.color_palette("hls", 8)[6]
plt.fill_between(range(T), 0, die_['code'], color=color_, alpha=0.2)
alpha_ = pd.DataFrame(gamma)
alpha_.iloc[:, 1].plot(color=color_, linewidth=2)
plt.xlabel("# trials")
plt.ylabel("probability of loaded dice")
plt.title('Output of backwards')
plt.show()


# In[274]:


# plot the estimated alpha and actual hidden states
color_ = sns.color_palette("hls", 8)[6]
alpha_ = pd.DataFrame(q_t)
plt.plot(alpha_, color=color_, linewidth=2)
plt.fill_between(range(T), 0, die_['code'], color=color_, alpha=0.2)
plt.xlabel("# trials")
plt.ylabel("probability of loaded dice")
plt.title('Output of viterbi')
plt.show()


# In[ ]:





# In[ ]:




