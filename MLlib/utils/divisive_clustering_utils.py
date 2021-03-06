from MLlib.models import KMeansClustering
import numpy as np
import matplotlib.pyplot as plt 
import math

def runKMeans(M, epochs):
    """
    
    """
    KMC = KMeansClustering()
    KMC.work(M, 2, epochs, print_flag=False)
    runs = 1
    while len(KMC.cluster_array[0]) == 0 or len(KMC.cluster_array[1]) == 0:
        KMC.work(M, 2, epochs, print_flag=False)  
        runs += 1
        if runs == 10:
            break
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

def to_adjacency_matrix(global_centroids: np.ndarray, n_clusters) -> np.ndarray:
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

def visualize(global_clusters, global_centroids, n_clusters, datasize):
    plt.scatter(global_centroids[:,0], global_centroids[:,1], color = 'red')
    for i in range(n_clusters):
        plt.annotate(f'c{i}', (global_centroids[i,0], global_centroids[i,1]))
    # plt.pause(2)
    plt.xlim((0, datasize))
    plt.ylim((0, datasize))
    '''
    dendrogram using adjacency matrix of the distances of the centroids from each other
    '''
    locations = []
    levels = []
    centroid_dist_mat = to_adjacency_matrix(global_centroids, n_clusters)
    # print(f'centroid_dist_mat: \n{centroid_dist_mat}\n')
    for i in range(n_clusters - 1):
        centroid_dist_mat, tup, dist = update_mat(centroid_dist_mat)
        # print('==========================================================')
        # print(f'updated matrix at {i}th iteration: \n{centroid_dist_mat}')
        locations.append(tup)
        levels.append(dist)
    # print(locations)
    # print(levels)
    label_map = create_label_map(locations, n_clusters)
    # print(label_map)
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
        # print('\t updating label_map[{0}]:{1}'.format(loc0,label_map[loc0]))

        ax.text(*c,label_map[loc0]['label'])

    _xticks=np.arange(0,n_clusters,1)
    # _xticklabels=['BA','NA','RM','FI','MI','TO']

    ax.set_xticks(_xticks)
    # ax.set_xticklabels(_xticklabels)

    ax.set_ylim(0,1.05*np.max(levels))
        
    plt.show()
