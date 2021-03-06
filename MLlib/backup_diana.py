'''
References: https://nlp.stanford.edu/IR-book/html/htmledition/divisive-clustering-1.html
https://www.geeksforgeeks.org/bisecting-k-means-algorithm-introduction/
Nice one: https://dmkd.cs.vt.edu/papers/kmeans13.pdf
http://www.philippe-fournier-viger.com/spmf/bisectingkmeans.pdf
Resources: https://github.com/scikit-learn/scikit-learn/issues/14214#issuecomment-507416805

'''
from MLlib.models import KMeansClustering
import numpy as np
import matplotlib.pyplot as plt 
import math

def run(M, epochs):
    """
    
    """
    KMC = KMeansClustering()
    KMC.work(M, 2, epochs, print_flag=False)
    while len(KMC.cluster_array[0]) == 0 or len(KMC.cluster_array[1]) == 0:
        KMC.work(M, 2, epochs, print_flag=False)  
    # KMC.cluster_array[0][0] = np.array(KMC.cluster_array[0][0])
    # KMC.cluster_array[1][0] = np.array(KMC.cluster_array[1][0])
    # KMC.centroid_array = np.array(KMC.centroid_array)
    # KMC.cluster_array = np.ndarray(, buffer=KMC.cluster_array)
    cluster_array = [np.empty((0,2)), np.empty((0, 2))] 
    for i, cluster in enumerate(KMC.cluster_array):
        for point in cluster:
            cluster_array[i] = np.append(cluster_array[i], point.reshape(1,2), axis=0)  

    # print(f'cluster_array: {cluster_array}')
    # print(f'cluster_array[0].shape: {cluster_array[0].shape}\ncluster_array[1].shape: {cluster_array[1].shape}')
    KMC.cluster_array = cluster_array
    return KMC.cluster_array, KMC.centroid_array

def distcalc(p1, p2):
    """
    Calculates the Euclidean Distance
    between p1 and p2 points.

    PARAMETERS
    ==========

    p1: ndarray(dtype=int,ndim=1,axis=1)
        Point array with its corresponding
        x and y coordinates.

    p2: ndarray(dtype=int,ndim=1,axis=1)
        Point array with its corresponding
        x and y coordinates.

    dist: int
        Sum of sqaurred difference of
        coordinates between p1 and p2
        points.

    distance: float
        Euclidean Distance between p1
        and p2 points.

    RETURNS
    =======

    distance: float
        Euclidean Distance between p1
        and p2 points.

    """
    dist = 0
    for i in range(0, len(p1)-1):
        dist += (p1[i]-p2[i])**2
    distance = dist**0.5
    return distance

def to_adjacency_matrix(global_centroids: np.ndarray) -> np.ndarray:
    '''
    Creates an adjacency matrix of the distances of the result centroids.
    '''
    centroid_dist_mat = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):
            centroid_dist_mat[i][j] = distcalc(global_centroids[i], global_centroids[j])
    for i in range(n_clusters):
        centroid_dist_mat[i,i] = np.nan
    return centroid_dist_mat

def update_mat(centroid_dist_mat):
    n_clusters=centroid_dist_mat.shape[0] 
    ind = np.unravel_index(np.nanargmin(centroid_dist_mat), (n_clusters, n_clusters))
    dist = centroid_dist_mat[ind]
    #ind is a tuple of x and y indices
    c = max(ind[0],ind[1])
    o = min(ind[0],ind[1])
    for i in range(n_clusters):
        a = centroid_dist_mat[c, i]
        b = centroid_dist_mat[o, i]
        m = 0.5 * math.sqrt(2*a*a + 2*b*b - dist*dist)
        centroid_dist_mat[o,i] = centroid_dist_mat[i,o] = m
    
    centroid_dist_mat[:,c] = centroid_dist_mat[c,:] = np.nan
    return centroid_dist_mat, ind, dist 

def create_label_map(locations, n_clusters):
    label_map = {}
    # last_free = n_clusters - 2
    xpos_map = np.array([False for i in range(n_clusters)])
    # print(f'xpos_map: {xpos_map}, shape: {xpos_map.shape}')
    last_free = n_clusters - 1
    for tup in locations:
        a, b = tup
        mn = min(a, b)
        mx = max(a,b)
        if mn not in label_map and mx not in label_map:
            label_map[mn] = {'label':f'c{mn}'}
            label_map[mn]['xpos'] = last_free
            xpos_map[last_free] = True
            label_map[mn]['ypos'] = 0
            last_free -= 1
            label_map[mx] = {'label':f'c{mx}'}
            label_map[mx]['xpos'] = last_free
            xpos_map[last_free] = True
            label_map[mx]['ypos'] = 0
            last_free -= 1
        elif mn in label_map and mx not in label_map:
            # print(f'label_map[mn]:{label_map[mn]}')
            label_map[mx] = {'label':f'c{mx}'}
            x = label_map[mn]['xpos']
            i = x - 1
            while xpos_map[i]:
                i -= 1
            label_map[mx]['xpos'] = i
            label_map[mx]['ypos'] = 0
        elif mx in label_map and mn not in label_map:
            # print(f'label_map[mx]:{label_map[mx]}')
            label_map[mn] = {'label':f'c{mn}'}
            x = label_map[mx]['xpos']
            i = x - 1
            # print(f'i={i}, x={x}')
            while xpos_map[i] and i>0:
                i -= 1
            label_map[mn]['xpos'] = i
            label_map[mn]['ypos'] = 0
    return label_map

def mk_fork(x0,x1,y0,y1,new_level):
    points=[[x0,x0,x1,x1],[y0,new_level,new_level,y1]]
    connector=[(x0+x1)/2.,new_level]
    return (points),connector

# X = np.genfromtxt('Examples/datasets/k_means_clustering.txt')
# print(f"X: {X}, type(X): {type(X)}")
X = np.empty((0, 2))
count = 1000
for i in range(count):
    x = np.random.randint(count) 
    y = np.random.randint(count) 
    X = np.append(X, [[x, y]], axis=0)
# print(len(X), len(X[0]), type(X), type(X[0]), type(X[0][0]))
# KMC = KMeansClustering()
n_clusters = 300
clusters, centroids = run(X, 7)
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
    clusters, centroids = run(parent, 7)
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

# plt.ion()
# for i in range(len(global_clusters)):
    # plt.scatter(global_clusters[i][:,0], global_clusters[i][:,1], color = ((i % 20) * 0.1, (i % 20) * 0.2, (i % 20) * 0.3, 1))
    # plt.scatter(global_clusters[i][:,0], global_clusters[i][:,1], color = 'black')
plt.scatter(global_centroids[:,0], global_centroids[:,1], color = 'red')
for i in range(n_clusters):
    plt.annotate(f'c{i}', (global_centroids[i,0], global_centroids[i,1]))
# plt.pause(2)
plt.xlim((0, 1000))
plt.ylim((0, 1000))
'''
dendrogram using adjacency matrix of the distances of the centroids from each other
'''
locations = []
levels = []
centroid_dist_mat = to_adjacency_matrix(global_centroids)
# print(f'centroid_dist_mat: \n{centroid_dist_mat}\n')
for i in range(n_clusters - 1):
    centroid_dist_mat, tup, dist = update_mat(centroid_dist_mat)
    # print('==========================================================')
    # print(f'updated matrix at {i}th iteration: \n{centroid_dist_mat}')
    locations.append(tup)
    levels.append(dist)
print(locations)
print(levels)
label_map = create_label_map(locations, n_clusters)
print(label_map)
#dendrogram code using locations and levels

fig,ax=plt.subplots()

for i,(new_level,(loc0,loc1)) in enumerate(zip(levels,locations)):

    # print('step {0}:\t connecting ({1},{2}) at level {3}'.format(i, loc0, loc1, new_level ))

    x0,y0=label_map[loc0]['xpos'],label_map[loc0]['ypos']
    x1,y1=label_map[loc1]['xpos'],label_map[loc1]['ypos']

    # print('\t points are: {0}:({2},{3}) and {1}:({4},{5})'.format(loc0,loc1,x0,y0,x1,y1))

    p,c=mk_fork(x0,x1,y0,y1,new_level)

    ax.plot(*p)
    ax.scatter(*c)

    # print('\t connector is at:{0}'.format(c))


    label_map[loc0]['xpos']=c[0]
    label_map[loc0]['ypos']=c[1]
    label_map[loc0]['label']='{0}/{1}'.format(label_map[loc0]['label'],label_map[loc1]['label'])
    print('\t updating label_map[{0}]:{1}'.format(loc0,label_map[loc0]))

    ax.text(*c,label_map[loc0]['label'])

_xticks=np.arange(0,n_clusters,1)
# _xticklabels=['BA','NA','RM','FI','MI','TO']

ax.set_xticks(_xticks)
# ax.set_xticklabels(_xticklabels)

ax.set_ylim(0,1.05*np.max(levels))
     
plt.show()

