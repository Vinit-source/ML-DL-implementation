import numpy as np
import matplotlib.pyplot as plt

class AgglomerativeClustering():
    def __init__(self):
        pass

    def initialize(self, dataset: matrix):
        self.dataset = dataset
        self.n = len(dataset)
        self.dist_mat = np.zeros((self.n, self.n))
    def compute_distance_matrix(self):
        for i in range(self.n):
            for j in i:
                self.dist_mat[i][j] = distance(self.cluster[i], self.dataset[j])


    def work(self, dataset):
        self.initialize(dataset)
        self.compute_distance_matrix(self)


