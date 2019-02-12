#Asher Bernardi - k-nearest neighbors
import numpy as np

def minkowski_L_k_norm(x, y, k):
    """Compute the Minkowski L_k norm between two vectors"""
    assert len(x) == len(y)
    return np.power( sum(abs(x[i] - y[i])**k for i in range(len(x))), 1./k)

def get_L_k_norm_function(k):
    """Returns a function that computes the L_k norm, given two vectors"""
    return lambda x,y: minkowski_L_k_norm(x, y, k)

def manhattan(x, y):
    """Compute the Manhattan distance (L1 norm)"""
    return minkowski_L_k_norm(x,y,1)

def euclidean(x, y):
    """Compute the euclidean distance (L2 norm)"""
    return minkowski_L_k_norm(x,y,2)

def single_knn(data, targets, k, metric, input):
    """Estimate the classification of a data point using k-nearest neighbors"""
    N = data.shape[0]
    # The distances from each of the N data points to x
    distances = [0]*N
    for j in range(N):
        distances[j] = metric(data[j],input)
    # zip the data together so it can sorted
    bundled_data = zip(data, targets, distances)
    bundled_data = sorted(bundled_data, key = lambda y: y[2])
    # Compute the bag of the classes: the number of elements in each class
    # found among x's k nearest neighboring elements.
    bag_classes = {}
    for j in range(k):
        if bundled_data[j][1] not in bag_classes:
            bag_classes[ bundled_data[j][1] ] = 0
        bag_classes[ bundled_data[j][1] ] += 1
    # find the max, that's the answer
    return max(bag_classes.items(), key = lambda y: y[1])[0]

def knn(data, targets, k, metric, inputs):
    """
    K-Nearest Neighbors
    """
    data = np.array(data)
    N = data.shape[0]
    D = data.shape[1]
    assert len(targets) == N
    assert k < N
    # For each input, we need to do the whole algorithm
    estimates = [0]*len(inputs)
    for i,x in enumerate(inputs):
        estimates[i] = single_knn(data, targets, k, metric, x)
    return estimates
