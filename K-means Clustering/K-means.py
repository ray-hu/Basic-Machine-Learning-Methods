#!/usr/bin/env python
# coding: utf-8

# In[113]:


import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.figure as figure


# In[114]:


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


# In[115]:


def centroid_update(points, closest, centroids):
    """returns the new centroids assigned from the points closest to them"""
    return np.array([points[closest==k].mean(axis=0) for k in range(centroids.shape[0])])


# In[116]:


def centroid_fit(points, centroids):
    """returns an array containing the index to the nearest centroid for each point"""
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0), np.amin(distances, axis=0)


# In[117]:


# initialization 
def init(X,K,init_type):
    if init_type == "1":
        idx = np.random.randint(len(X), size=K)
        init_u = X[idx]
    elif init_type =="2":
        init_u = X[np.random.randint(len(X), size=1)]
        while len(init_u) < K:
            closest, dist = centroid_fit(X,init_u)
            init_u = np.vstack((init_u,X[np.random.choice(len(X),1, p=dist/np.sum(dist))]))
    elif init_type == "3":
        init_u = np.zeros((K,X.shape[1]))
        for i in range(K):
            mask = testing_labels == i
            init_u[i] = X[np.random.randint(sum(mask), size=1)]
        
    return init_u


# In[118]:


def assign(K,X,init_type):
    c = init(X, K, init_type)
    i = 0
    while True:
        i += 1
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
    #print('Iteration = {}.'.format(i))
    return assignment, Final_centroid


# In[119]:


def show_output_1(assn,centroid):
    fig = plt.figure()
    ag = gridspec.GridSpec(ncols=5, nrows=2, figure=fig)
    ax1 = fig.add_subplot(ag[0, 0])
    ax1.imshow(centroid[0].reshape((28,28)))
    ax2 = fig.add_subplot(ag[0,1])
    ax2.imshow(centroid[1].reshape((28,28)))
    ax3 = fig.add_subplot(ag[0,2])
    ax3.imshow(centroid[2].reshape((28,28)))
    ax4 = fig.add_subplot(ag[0,3])
    ax4.imshow(centroid[3].reshape((28,28)))
    ax5 = fig.add_subplot(ag[0,4])
    ax5.imshow(centroid[4].reshape((28,28)))
    ax6 = fig.add_subplot(ag[1,0])
    ax6.imshow(centroid[5].reshape((28,28)))
    ax7 = fig.add_subplot(ag[1,1])
    ax7.imshow(centroid[6].reshape((28,28)))
    ax8 = fig.add_subplot(ag[1,2])
    ax8.imshow(centroid[7].reshape((28,28)))
    ax9 = fig.add_subplot(ag[1,3])
    ax9.imshow(centroid[8].reshape((28,28)))
    ax10 = fig.add_subplot(ag[1,4])
    ax10.imshow(centroid[9].reshape((28,28)))
    plt.show()


# In[120]:


def show_output(assignment, centroid):
    
    img1 = X[assignment[:,0]==1][np.random.choice(np.int(sum(assignment[:,0])),4)]
    img2 = X[assignment[:,1]==1][np.random.choice(np.int(sum((assignment[:,1]))),4)]
    img3 = X[assignment[:,2]==1][np.random.choice(np.int(sum((assignment[:,2]))),4)]
    fig = plt.figure()
    ag = gridspec.GridSpec(ncols=4, nrows=6, figure=fig)
    ax1 = fig.add_subplot(ag[0:2, 0:2])
    ax1.imshow(centroid[0].reshape((28,28)))
    ax1_1 = fig.add_subplot(ag[0,2])
    ax1_1.imshow(img1[0].reshape((28,28)))
    ax1_2 = fig.add_subplot(ag[0,3])
    ax1_2.imshow(img1[1].reshape((28,28)))
    ax1_3 = fig.add_subplot(ag[1,2])
    ax1_3.imshow(img1[2].reshape((28,28)))
    ax1_4 = fig.add_subplot(ag[1,3])
    ax1_4.imshow(img1[3].reshape((28,28)))
    ax2 = fig.add_subplot(ag[2:4,0:2])
    ax2.imshow(centroid[1].reshape((28,28)))
    ax2_1 = fig.add_subplot(ag[2,2])
    ax2_1.imshow(img2[0].reshape((28,28)))
    ax2_2 = fig.add_subplot(ag[2,3])
    ax2_2.imshow(img2[1].reshape((28,28)))
    ax2_3 = fig.add_subplot(ag[3,2])
    ax2_3.imshow(img2[2].reshape((28,28)))
    ax2_4 = fig.add_subplot(ag[3,3])
    ax2_4.imshow(img2[3].reshape((28,28)))
    ax3 = fig.add_subplot(ag[4:6,0:2])
    ax3.imshow(centroid[2].reshape((28,28)))
    ax3_1 = fig.add_subplot(ag[4,2])
    ax3_1.imshow(img3[0].reshape((28,28)))
    ax3_2 = fig.add_subplot(ag[4,3])
    ax3_2.imshow(img3[1].reshape((28,28)))
    ax3_3 = fig.add_subplot(ag[5,2])
    ax3_3.imshow(img3[2].reshape((28,28)))
    ax3_4 = fig.add_subplot(ag[5,3])
    ax3_4.imshow(img3[3].reshape((28,28)))
    plt.show()


# In[121]:


#
K=10
assignment_1, centroid_1 = assign(K,X,init_type="1")
show_output_1(assignment_1, centroid_1)


# In[ ]:


assignment_2, centroid_2 = assign(K,X,init_type="2")
show_output_1(assignment_2, centroid_2)


# In[71]:


assignment_3, centroid_3 = assign(K,X,init_type="3")
show_output_1(assignment_3, centroid_3)


# In[72]:


# 
K=3
assignment, centroid = assign(K,X,init_type="3")
show_output(assignment, centroid)
