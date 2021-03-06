'''
References: https://nlp.stanford.edu/IR-book/html/htmledition/divisive-clustering-1.html
https://www.geeksforgeeks.org/bisecting-k-means-algorithm-introduction/
Nice one: https://dmkd.cs.vt.edu/papers/kmeans13.pdf
http://www.philippe-fournier-viger.com/spmf/bisectingkmeans.pdf
Resources: https://github.com/scikit-learn/scikit-learn/issues/14214#issuecomment-507416805

'''
from MLlib.utils.divisive_clustering_utils import runKMeans, visualize
import numpy as np

class DivisiveClustering:
    def work(self, M, n_clusters, n_iterations):
        clusters, centroids = runKMeans(X, 7)
        # print(f"type(centroids):{type(KMC.centroid_array)}\n centroid_array: {KMC.centroid_array}\n type(centroid_array[0]): {type(KMC.centroid_array[0])}")
        # global_clusters = list(clusters)
        # global_centroids = list(centroids)
        global_clusters = clusters
        global_centroids = centroids
        # print(f'global_clusters: {global_clusters}, \nglobal_centroids: {global_centroids}')
        # print(f'(len(c1), len(c2)): {(len(clusters[0]), len(clusters[1]))}')

        i = 2
        while i < n_clusters:
            max_len = 0
            rem_index = -1
            for j, cluster in enumerate(global_clusters):
                l = len(cluster)
                if l > max_len:
                    max_len = l
                    rem_index = j
            parent = global_clusters[rem_index]
            del(global_clusters[rem_index]) 
            # print(f"Before delete: {global_centroids}")
            del(global_centroids[rem_index]) 
            # print(f"After delete: {global_centroids}")
            # print(f'parent: {parent}')
            # global_clusters = np.delete(global_clusters, rem_index, 0)
            # global_centroids = np.delete(global_centroids, rem_index, 0)
            clusters, centroids = runKMeans(parent, 7)
            global_clusters.append(clusters[0])
            global_clusters.append(clusters[1])
            global_centroids.append(centroids[0])
            global_centroids.append(centroids[1])
            # global_clusters = np.append(global_clusters, clusters[0])
            # global_clusters = np.append(global_clusters, clusters[1])
            # global_centroids = np.append(global_centroids, centroids[0])
            # global_centroids = np.append(global_centroids, centroids[1])
            i += 1

        # print(f'global_centroids: {global_centroids}')
        # print(f'global_clusters: {global_clusters}')

        global_centroids = np.array(global_centroids)
        return global_clusters, global_centroids

    # def work(self, M, n_clusters, n_iterations):
    #     clusters, centroids = runKMeans(M, n_iterations)
    #     # print(f"type(centroids):{type(KMC.centroid_array)}\n centroid_array: {KMC.centroid_array}\n type(centroid_array[0]): {type(KMC.centroid_array[0])}")
    #     # global_clusters = list(clusters)
    #     # global_centroids = list(centroids)
    #     global_clusters = clusters
    #     global_centroids = centroids
    #     # print(f'global_clusters: {global_clusters}, \nglobal_centroids: {global_centroids}')
    #     # print(f'(len(c1), len(c2)): {(len(clusters[0]), len(clusters[1]))}')

    #     i = 2
    #     while i < n_clusters:
    #         max_len = 0
    #         rem_index = -1
    #         for j, cluster in enumerate(global_clusters):
    #             l = len(cluster)
    #             if l > max_len:
    #                 max_len = l
    #                 rem_index = j
    #         parent = global_clusters[rem_index]
    #         del(global_clusters[rem_index]) 
    #         # print(f"Before delete: {global_centroids}")
    #         del(global_centroids[rem_index]) 
    #         # print(f"After delete: {global_centroids}")
    #         # print(f'parent: {parent}')
    #         # global_clusters = np.delete(global_clusters, rem_index, 0)
    #         # global_centroids = np.delete(global_centroids, rem_index, 0)
    #         clusters, centroids = runKMeans(parent, n_iterations)
    #         global_clusters.append(clusters[0])
    #         global_clusters.append(clusters[1])
    #         global_centroids.append(centroids[0])
    #         global_centroids.append(centroids[1])
    #         # global_clusters = np.append(global_clusters, clusters[0])
    #         # global_clusters = np.append(global_clusters, clusters[1])
    #         # global_centroids = np.append(global_centroids, centroids[0])
    #         # global_centroids = np.append(global_centroids, centroids[1])
    #         i += 1
    #     global_centroids = np.array(global_centroids)
    #     return global_clusters, global_centroids

X = np.genfromtxt('Examples/datasets/k_means_clustering.txt')
count = X.shape[0]
# print(f"X: {X}, type(X): {type(X)}")
# X = np.empty((0, 2))
# count = 1000
# for i in range(count):
#     x = np.random.randint(count) 
#     y = np.random.randint(count) 
#     X = np.append(X, [[x, y]], axis=0)
# print(len(X), len(X[0]), type(X), type(X[0]), type(X[0][0]))
# KMC = KMeansClustering()
n_clusters = 3
DC = DivisiveClustering()
global_clusters, global_centroids = DC.work(X, n_clusters, 7)
visualize(global_clusters, global_centroids, n_clusters, count)
# print(f'global_centroids: {global_centroids}')
# print(f'global_clusters: {global_clusters}')



# plt.ion()
# for i in range(len(global_clusters)):
    # plt.scatter(global_clusters[i][:,0], global_clusters[i][:,1], color = ((i % 20) * 0.1, (i % 20) * 0.2, (i % 20) * 0.3, 1))
    # plt.scatter(global_clusters[i][:,0], global_clusters[i][:,1], color = 'black')
# print(label_map)
#dendrogram code using locations and levels
# display_dendrogram(locations, levels, label_map)

