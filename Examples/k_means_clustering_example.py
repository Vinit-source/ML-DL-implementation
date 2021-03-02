from MLlib.models import KMeansClustering
import numpy as np
import matplotlib.pyplot as plt 

X = np.genfromtxt('datasets/k_means_clustering.txt')
# X = np.empty((0, 2))
# for i in range(10000):
#     x = np.random.randint(10000) 
#     y = np.random.randint(10000) 
#     X = np.append(X, [[x, y]], axis=0)
# print(len(X), len(X[0]), type(X), type(X[0]), type(X[0][0]))
KMC = KMeansClustering()
KMC.work(X, 2, 7)
