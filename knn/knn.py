"""
@ knn  lazy learning
@ two ways: general searching & kdtree searching
@ wttttt at 2016.12.07
"""
import sys
import numpy as np
import pandas as pd
import time
start = time.clock()


def load_data(train_file, test_file):
    train = pd.read_csv(train_file)  # the first col is id, last col is label
    test = pd.read_csv(test_file)    # the first col is id
    y = train.iloc[:, -1]
    train = train.drop(labels= train.columns[-1], axis=1)
    return np.array(train), np.array(y), np.array(test)


def do_classify(k=10, train_file='train.csv', test_file='test.csv'):
    train, y, test = load_data(train_file, test_file)
    num_instance, num_cols = train.shape
    train = train[:, 1:]  # removing id
    y_test = []   # storing the y of testing data
    for test_one in test:  # for each testing instances
        print 'predicting for test id{0}'.format(test_one[0])
        test_one = test_one[1:]   # removing id
        # compute the diff of the test instance of each train instance
        diff = train - np.tile(test_one, (num_instance, 1))
        squre_diff = np.square(diff)
        distance = np.sum(squre_diff, axis=1)**0.5   # the square distance
        topk_index = np.argsort(distance)
        classCounter = {}
        for i in range(k):
            classCounter[y[topk_index[i]]] = classCounter.get(y[topk_index[i]],0) + 1
        y_test.append(sorted(classCounter)[0])
        print 'predicted: y is {0}'.format(y_test[-1])
    print 'all prediction is done, writing...'
    with open('result_knn.csv', 'w') as fi:
        for i in range(len(y_test)):
            fi.write(('{0},{1}\n').format(test[i, 0], y_test[i]))


do_classify(k=int(sys.argv[1]), train_file=sys.argv[2], test_file=sys.argv[3])
end = time.clock()
print "time: %f s" % (end - start)



