"""
@ knn from scratch, including kdtree
@ k-nearest-neigbors algo using kdtree to search
@ wttttt at 2016.12.13-15
@ github, see https://github.com/wttttt-wang
"""
import kdtree
import pandas as pd
import numpy as np
import itertools
import math
import sys
import time
start = time.clock()


k = sys.argv[1]
train_file = sys.argv[2]
test_file = sys.argv[3]

# step1: reading data
def load_data(train_file, test_file):
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    y = train.iloc[:, -1]
    train = train.drop(labels=train.columns[-1], axis=1)
    return np.array(train), np.array(y), np.array(test)
train, y, test = load_data(train_file, test_file)
train = train[:, 1:]  # removing id
test_id = test[:, 0]
test = test[:, 1:]  # removing id

# step2: constructig kdtree for training data
tree = kdtree.KdTree(n_dim=train.shape[1])
tree.createTree(np.c_[train, y])

def clear_node(root):
    """
    the function used to clear all the nodes's flag to 0
    root: the root node of a tree
    """
    if root is not None:
        root.flag = 0
    if root.left is not None:
        clear_node(root.left)
    if root.right is not None:
        clear_node(root.right)
        
# vote for prediction
y_test = []   # storing the y of testing data
for i in range(test.shape[0]):
    print 'predicting for test id {0}'.format(test_id[i])
    classCounter = {}  # vote
    dis, k_nearest = tree.k_nearest_neighbor(k, test[i], tree.root, kdtree.LargeHeap())
    for pos in k_nearest:
        classCounter[pos[-1]] = classCounter.get(pos[-1], 0) + 1
    y_test.append(sorted(classCounter)[0])
    print 'predicted: y is {0}'.format(y_test[-1])
    clear_node(tree.root)   # clear the tree's nodes's flag after each search
print 'all prediction is done, writing...'
with open('result_knn_kdtree_fromscratch.csv','w') as fi:
    for i in range(len(y_test)):
        fi.write(('{0},{1}\n').format(test_id[i], y_test[i]))

end = time.clock()
print "time: %f s" % (end - start)
