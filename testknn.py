# author: Asher Bernardi
import numpy as np
import knn
from sklearn import datasets as ds
from random import shuffle
import time
from prettytable import PrettyTable
from sklearn import neighbors

def testknn(name, data, targets, test_ratios, ks, metric):
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
            filename = 'errorlogs/{0:s}_errorlog_t={1:.2f}_k={2:d}'.format(name,t,k)
            test_results[i][j] = compare_lists(estimates,list(targets[cutoff:]), filename)

    # print out results
    result_table = PrettyTable()
    result_table.field_names = ["test \\ k"] + ([str(k) for k in ks])
    for i,t in enumerate(test_ratios):
        result_table.add_row(['{0:d}%'.format(int(t*100))] + (['{0:.2f}%'.format(test_results[i][j] * 100) for j in range(len(ks))]))
    print(result_table)

def compare_lists(list1, list2, filename):
    '''find the percentage of agreement between two lists'''
    N = len(list1)
    assert len(list2) == N
    file = open(filename, 'w+')

    num_agreed = 0
    for i in range(N):
        if list1[i] == list2[i]:
            num_agreed += 1
        else:
            file.write(str(list1[i]) + ' =/= ' + str(list2[i]) + '\n')
    file.close()

    return num_agreed / N

def parseCSV(filename):
    """Parse a CSV file into an ndarray"""
    file = open(filename)
    array = []
    for line in file:
        #the [:-1] is to ignore the newline at the end of each line
        array.append(line[:-1].split(','))
    file.close()
    return np.array(array)

# ***** Now we do the testing! *****

# Sci-Kit learn's Iris dataset
iris = ds.load_iris()
print('\n-> Results from running knn on the "iris" dataset using Euclidean distance\n')
now = time.time()
testknn('iris', iris.data, iris.target, [0.05,0.15,0.25], [5,10,15], knn.euclidean)
print("\nRunning time (sec): {0:f}\n".format(time.time() - now))

# Sci-Kit learn's Wine dataset
wine = ds.load_wine()
print('\n-> Results from running knn on the "wine" dataset using Manhattan distance:\n')
now = time.time()
testknn('wineManhattan', wine.data, wine.target, [0.10,0.20,0.30], [5,10,15], knn.manhattan)
print("\nRunning time (sec): {0:f}\n".format(time.time() - now))

# Sci-Kit learn's Wine dataset
wine = ds.load_wine()
print('\n-> Results from running knn on the "wine" dataset using L_13 norm:\n')
now = time.time()
testknn('wineL13', wine.data, wine.target, [0.10,0.20,0.30], [5,10,15], knn.get_L_k_norm_function(13))
print("\nRunning time (sec): {0:f}\n".format(time.time() - now))

# Wine dataset using sklearn
sklknn = neighbors.KNeighborsClassifier()
bundle = list(zip(wine.data[:300], wine.target[:300]))
shuffle(bundle)
wine_data, wine_target = zip(*bundle)
print('\n-> Results from running sklearn\'s knn on the "wine" dataset:\n')
now = time.time()
sklknn.fit(wine_data[:151], wine_target[:151])
print('{0:.2f}%'.format(100 * sklknn.score(wine_data[151:], wine_target[151:])))
print("\nRunning time (sec): {0:f}\n".format(time.time() - now))

# UCI's mushroom dataset found at
# https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/
mushroom = parseCSV('test/agaricus-lepiota.data')
# We use only use 3000 of the mushroom data points so it won't take so long
# We also will convert the letters in the data to ascii so it can quantified
shuffle(mushroom)
mushroom_targets = [ord(d[0]) for d in mushroom[:3000]]
mushroom_data = [[ord(a) for a in d[1:]] for d in mushroom[:3000]]
print('\n-> Results from running knn on the "mushroom" dataset using Manhattan distance:\n')
now = time.time()
testknn('mushroom', mushroom_data, mushroom_targets, [0.15], [10], knn.manhattan)
print('\nRunning time (sec): {0:f}\n'.format(time.time() - now))

# Sci-Kit Learn's Digits dataset
digits = ds.load_digits()
print('\n-> Results from running knn on sklearn\'s "digits" dataset using Euclidean distance:\n')
now = time.time()
testknn('SKLdigits', digits.data, digits.target, [0.10], [10], knn.euclidean)
print("\nRunning time (sec): {0:f}\n".format(time.time() - now))

# My own hand-written digits dataset
digits_data = parseCSV('test/digits.data')
digits_target = parseCSV('test/digits.target')
digits_data = np.array([[int(j) for j in digits_data[i]] for i in range(len(digits_data))])
digits_target = np.array([int(i) for i in digits_target])
print('\n-> Results from running knn on my own digits dataset using Manhattan distance:\n')
now = time.time()
testknn('myDigits', digits_data[:300], digits_target[:300], [0.15], [5], knn.manhattan)
print("\nRunning time (sec): {0:f}\n".format(time.time() - now))

# My own hand-written digits dataset using sklearn
sklknn = neighbors.KNeighborsClassifier()
bundle = list(zip(digits_data[:300], digits_target[:300]))
shuffle(bundle)
digits_data, digits_target = zip(*bundle)
print('\n-> Results from running sklearn\'s knn on my own digits dataset distance:\n')
now = time.time()
sklknn.fit(digits_data[:255], digits_target[:255])
print('{0:.2f}%'.format(100 * sklknn.score(digits_data[255:300], digits_target[255:300])))
print("\nRunning time (sec): {0:f}\n".format(time.time() - now))

# My own hand-written digits dataset with size 16
digits_data = parseCSV('test/digits16.data')
digits_target = parseCSV('test/digits16.target')
digits_data = np.array([[int(j) for j in digits_data[i]] for i in range(len(digits_data))])
digits_target = np.array([int(i) for i in digits_target])
print('\n-> Results from running knn on my own digits dataset with size 16 using Euclidean distance:\n')
now = time.time()
testknn('myDigits16', digits_data[:300], digits_target[:300], [0.15], [5], knn.euclidean)
print("\nRunning time (sec): {0:f}\n".format(time.time() - now))

# My own hand-written digits dataset with size 16 using sklearn
sklknn = neighbors.KNeighborsClassifier()
bundle = list(zip(digits_data[:300], digits_target[:300]))
shuffle(bundle)
digits_data, digits_target = zip(*bundle)
print('\n-> Results from running sklearn\'s knn on my own digits with size 16 dataset distance:\n')
now = time.time()
sklknn.fit(digits_data[:255], digits_target[:255])
print('{0:.2f}%'.format(100 * sklknn.score(digits_data[255:300], digits_target[255:300])))
print("\nRunning time (sec): {0:f}\n".format(time.time() - now))

print("Testing Complete!\n")