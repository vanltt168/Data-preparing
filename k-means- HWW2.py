#!/usr/bin/env python
# coding: utf-8

# In[38]:


from __future__ import print_function 
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
np.random.seed(11)


# In[39]:


means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

X = np.concatenate((X0, X1, X2), axis = 0)
K = 3

original_label = np.asarray([0]*N + [1]*N + [2]*N).T


# In[40]:


def kmeans_display(X, label):
    K = np.amax(label) + 1
    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]
    
    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize = 4, alpha = .8)
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize = 4, alpha = .8)
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize = 4, alpha = .8)

    plt.axis('equal')
    plt.plot()
    plt.show()
    
kmeans_display(X, original_label)


# In[45]:


def kmeans_init_centers(X, k):
    # randomly pick k rows of X as initial centers
    return X[np.random.choice(X.shape[0], k, replace=False)]

def kmeans_assign_labels(X, centers):
    D = cdist(X, centers)
    return np.argmin(D, axis = 1)

def kmeans_update_centers(X, labels, K):
    centers = np.zeros((K, X.shape[1]))
    for k in range(K):
        Xk = X[labels == k, :]
        # take average
        centers[k,:] = np.mean(Xk, axis = 0)
    return centers

def has_converged(centers, new_centers):
    return (set([tuple(a) for a in centers]) == 
        set([tuple(a) for a in new_centers]))


# In[46]:


def kmeans_visualize(X, centers, labels, n_cluster, title):
    plt.xlabel('x') 
    plt.ylabel('y') 
    plt.title(title) 
    plt_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'] 
 
    for i in range(n_cluster):
        data = X[labels == i] 
        plt.plot(data[:, 0], data[:, 1], plt_colors[i] + '^', markersize = 4, label = 'cluster_' + str(i)) # Vẽ cụm i lên đồ thị
        plt.plot(centers[i][0], centers[i][1],  plt_colors[i+4] + 'o', markersize = 10, label = 'center_' + str(i)) # Vẽ tâm cụm i lên đồ thị
        plt.legend() 
    plt.show()


# In[58]:


def kmeans(init_centes, init_labels, X,K):
    centers = init_centes
    labels = init_labels
    times = 0
    while True:
        labels = kmeans_assign_labels(X, centers)
        kmeans_visualize(X, centers, labels,K, 'Assigned label for data at time = ' + str(times + 1))
        new_centers = kmeans_update_centers(X, labels, K)
        if has_converged(centers, new_centers):
              break
        centers = new_centers
        kmeans_visualize(X, centers, labels, K, 'at time = ' + str(times + 1))
        times += 1
    return (centers, labels, times)


# In[57]:


init_centers = kmeans_init_centers(X, K)
print(init_centers) 
init_labels = np.zeros(X.shape[0])
kmeans_visualize(X, init_centers, init_labels, K, 'Assigned all data as cluster 0')
centers, labels, times = kmeans(init_centers, init_labels, X,K)
 
print('Done! Kmeans has converged after', times, 'times')


# In[ ]:





# In[ ]:





# In[ ]:




