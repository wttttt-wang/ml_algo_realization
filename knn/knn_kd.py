"""
@ knn  lazy learning
@ kdtree for k-nearest-neighbor searching
@ wttttt at 2016.12.12
"""

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import sys
import time
start = time.clock()


k = int(sys.argv[1])
train_file = sys.argv[2]
test_file = sys.argv[3]

def load_data(train_file, test_file):
    train = pd.read_csv(train_file)  # the first col is id, last col is label
    test = pd.read_csv(test_file)    # the first col is id
    y = train.iloc[:, -1]
    train = train.drop(labels= train.columns[-1], axis=1)
    return np.array(train), np.array(y), np.array(test)

train, y, test = load_data(train_file, test_file)
train = train[:, 1:]  # removing id
test_id = test[:, 0]
test = test[:,1:]   # removing id

# step2: constructig kdtree for training data
tree = KDTree(train)

#find the k nearest neighbor
dis, nearest_loc = tree.query(x=test, k=k, p=2) # p=2 means Euclidean distance

# vote for prediction
y_test = []   # storing the y of testing data
for i in range(nearest_loc.shape[0]):
    print 'predicting for test id {0}'.format(test_id[i])
    classCounter = {}  # vote
    for pos in nearest_loc[i]:
        classCounter[y[pos]] = classCounter.get(y[pos], 0) + 1
    y_test.append(sorted(classCounter)[0])
    print 'predicted: y is {0}'.format(y_test[-1])
print 'all prediction is done, writing...'
with open('result_knn_kdtree.csv') as fi:
    for i in range(len(y_test)):
        fi.write(('{0},{1}\n').format(test_id[i], y_test[i]))

end = time.clock()
print "time: %f s" % (end - start)


