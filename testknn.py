# author: Asher Bernardi
import numpy as np
import knn
from sklearn import datasets as ds
from random import shuffle
import time
from prettytable import PrettyTable

def testknn(data, targets, test_ratios, ks, metric):
    # shuffle the data to ensure an even spread of classes
    pairs = list(zip(data, targets))

    N = len(targets)
    # test many times using each of the test_ratios and numbers for k
    test_results = np.zeros((len(test_ratios), len(ks)))
    for i,t in enumerate(test_ratios):
        for j, k in enumerate(ks):
            shuffle(pairs)
            data, targets = zip(*pairs)
            # cutoff is the index in the data where we will distinguish between
            # training and testing data
            cutoff = int((1-t) * N)
            estimates = knn.knn(data[:cutoff], targets[:cutoff], k, metric, data[cutoff:])
            test_results[i][j] = compare_lists(estimates,list(targets[cutoff:]))

    # print out results
    result_table = PrettyTable()
    result_table.field_names = ["test \\ k"] + ([str(k) for k in ks])
    for i,t in enumerate(test_ratios):
        result_table.add_row(['{0:d}%'.format(int(t*100))] + (['{0:.2f}%'.format(test_results[i][j] * 100) for j in range(len(ks))]))
    print(result_table)

def compare_lists(list1, list2):
    '''find the percentage of agreement between two lists'''
    N = len(list1)
    assert len(list2) == N

    num_agreed = 0
    for i in range(N):
        if list1[i] == list2[i]: num_agreed += 1

    return num_agreed / N

def parseCSV(file):
    """Parse a CSV file into an ndarray"""
    array = []
    for line in file:
        #the [:-1] is to ignore the newline at the end of each line
        array.append(line[:-1].split(','))
    return np.array(array)

# Now we do the testing!
# Sci-Kit learn's Iris dataset
iris = ds.load_iris()
print('\nResults from running knn on the Iris dataset using Euclidean distance\n')
now = time.time()
testknn(iris.data, iris.target, [0.05,0.15,0.25], [5,10,15], knn.euclidean)
print("\nRunning time (sec): {0:f}\n".format(time.time() - now))

# Sci-Kit learn's Wine dataset
wine = ds.load_wine()
print('\nResults from running knn on the Wine dataset using Manhattan distance:\n')
now = time.time()
testknn(wine.data, wine.target, [0.10,0.20,0.30], [10,15,20], knn.manhattan)
print("\nRunning time (sec): {0:f}\n".format(time.time() - now))

# Sci-Kit learn's Wine dataset
wine = ds.load_wine()
print('\nResults from running knn on the Wine dataset using Euclidean distance:\n')
now = time.time()
testknn(wine.data, wine.target, [0.10,0.20,0.30], [10,15,20], knn.euclidean)
print("\nRunning time (sec): {0:f}\n".format(time.time() - now))

# UCI's mushroom dataset found at
# https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/
mushroom = parseCSV(open('test/agaricus-lepiota.data'))
# We use only use 3000 of the mushroom data points so it won't take so long
shuffle(mushroom)
mushroom_targets = [ord(d[0]) for d in mushroom[:3000]]
mushroom_data = [[ord(a) for a in d[1:]] for d in mushroom[:3000]]
print('\nResults from running knn on the Mushroom dataset using Manhattan distance:\n')
now = time.time()
testknn(mushroom_data, mushroom_targets, [0.15], [10], knn.manhattan)
print('\nRunning time (sec): {0:f}\n'.format(time.time() - now))

# Sci-Kit learn's Digits dataset
digits = ds.load_digits()
print('\nResults from running knn on the Digits dataset using Euclidean distance:\n')
now = time.time()
testknn(digits.data, digits.target, [0.10,0.20,0.30], [10,15,20], knn.euclidean)
print("\nRunning time (sec): {0:f}\n".format(time.time() - now))

