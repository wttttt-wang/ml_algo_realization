# -*- coding:utf-8 -*- -
"""
@ NaiveBayes from scratch
@ wttttt at 2016.12.21
@ github, see https://github.com/wttttt-wang/ml_algo_realization
@ NaiveBayes, p(y|x) = (p(x|y)*p(y))/p(x) = ((∏p(xi|y))*p(y))/p(x).
    for a fixed x, we only need to calculate (∏p(xi|y))*p(y))
    !! addition: use log transform
    @ update20170227: to use log transform, u must do some smoothing ==> because log(0) is illegal
    !! smoothing_para->alpha ==>p(xi|y) = (I(xi)+alpha])/(I+alpha) (laplace smoothing when alpha==1)
    for more information see https://en.wikipedia.org/wiki/Naive_Bayes_classifier
@ About this realization:
    1) use pickle to pick the already trained model
    2) use log transformation to avoid overflow(because of multiplication of many small p(xi|y))
    3) handle two kinds of data. one is basic train/test samples; the other is text.
    4) multi-class classification, instead of 0/1 classification
@ ToDo: some advanced smoothing?-?
"""
import numpy as np
import pandas as pd
import os
import pickle
import math
import time
start = time.clock()


def load_data(train_file):
    '''
    :param train_file: type str. including a header in this csv file
    :param test_file: type str. including a header in this csv file
    :return:
        1) train features: type array. not include the label. but including id
        2) train label: type array
        3) test features: type array. including id
    '''
    train = pd.read_csv(train_file)  # the first col is id, last col is label. including header in this file.
    # test = pd.read_csv(test_file)    # the first col is id. including header in this file.
    y = train.iloc[:, -1]
    train = train.drop(labels=train.columns[-1], axis=1)
    # return np.array(train), np.array(y), np.array(test)
    return np.array(train), np.array(y)


def load_text(train_file, test_file):
    # TODO: here
    pass


def train_model(train, y, job_name, alpha=0):
    """
    :param train: type array, not including id
    :param y: type array
    :param job_name: the job name which u can assigned to distinguish from other job ==> used in the name of ouputfile
    :param alpha: type float, parameter for smoothing. 0 for no smoothing
    :return px_y: type dict of list of dict
    :return py: type dict
    """
    # calculate p(xi|yj), use dic(dic of list of dic) to store this&& calculate p(yi)
    # p(xi|yj) = dic[yj][i][xi]
    num_sample, num_feature = train.shape

    px_y = {}
    sum_pxy = {}
    default_pxy = {}
    for y_value in set(y):
        px_y[y_value] = [{} for i in range(num_feature)]
        sum_pxy[y_value] = [alpha for i in range(num_feature)]

    # use dict to store p(y)
    py = {}
    # calculate p(x|y) & p(y)

    for i in range(num_sample):
        py[y[i]] = py.get(y[i], 0) + 1
        for j in range(num_feature):
            px_y[y[i]][j][train[i][j]] = px_y[y[i]][j].get(train[i][j], alpha) + 1
            sum_pxy[y[i]][j] += 1

    for key in py:
        py[key] /= float(len(y))
        default_pxy[key] = alpha / (alpha + py[key] * len(y))
    for y_key in px_y:
        # handle each y
        for i in range(len(px_y[y_key])):
            # handle each feature
            for fea_key in px_y[y_key][i]:
                # handle each feature's value
                px_y[y_key][i][fea_key] /= float(sum_pxy[y_key][i])

    # pickle the already trained model here.
    if os.path.exists('./output') is False:
        !mkdir. / output
    file_name = './output/model_nb_' + job_name
    with open(file_name, 'w') as fi:
        pickle.dump(px_y, fi)
        pickle.dump(py, fi)
        pickle.dump(default_pxy, fi)
        # pickle.dump(p_fea_label, fi)
        # return px_y, py


def classify(train_file, test_file, job_name, alpha=0):
    """
    :param train_file:
    :param test_file:
    :param job_name: the job name which u can assigned to distinguish from other job ==> used in the name of ouputfile
    :return:
    """
    test = pd.read_csv(test_file)  # the first col is id. including header in this file.
    test = np.array(test)
    file_name = './output/model_nb_' + job_name
    if os.path.exists(file_name) is False:
        train, y = load_data(train_file)
        train_model(train, y, job_name, alpha)
    pk_file = open(file_name, 'rb')
    px_y = pickle.load(pk_file)
    py = pickle.load(pk_file)
    default_pxy = pickle.load(pk_file)
    pk_file.close()

    y_test = []  # storing the y of testing data

    for test_one in test:
        print 'predicting for test id{0}'.format(test_one[0])
        test_one = test_one[1:]  # removing id
        # compute each p(y|x), then choose the bigger one to be the predicted y
        pxy_max = 0
        for tag_y in px_y:
            # p(x1|y)p(x2|y)...p(y)
            # !!!!log transform  ==> to avoid to small value
            pxy = math.log(py[tag_y])  # p(y)
            for i in range(len(px_y[tag_y])):
                # pxy += math.log(px_y[tag_y][i][test_one[i]])
                pxy += math.log(px_y[tag_y][i].get(test_one[i], default_pxy[tag_y]))
            if pxy > pxy_max:
                bigger_tag = tag_y  # the chosen tag for current test_one
                pxy_max = pxy
        y_test.append(tag_y)
        print 'predicted: y is {0}'.format(y_test[-1])

    print 'all prediction is done, writing...'
    if os.path.exists('./output') is False:
        !mkdir ./output
    outputfile = './output/result_nb_' + job_name
    with open(outputfile, 'w') as fi:
        for i in range(len(y_test)):
            fi.write(('{0},{1}\n').format(test[i, 0], y_test[i]))


def test_nb():
    train_file = './data/tra_282.csv'
    test_file = './data/tes_282.csv'
    job_name = '0228_2'
    train, y = load_data(train_file)
    # print train_model(train, y)
    alpha = 1
    train_model(train, y, job_name, alpha)
    classify(train_file, test_file, job_name)
    print '-----probability-------'
    pk_file = open('./output/model_nb_0228_2', 'rb')
    data1 = pickle.load(pk_file)
    print data1
    data2 = pickle.load(pk_file)
    print data2
    pk_file.close()

'''
classify(train_file=sys.argv[1], test_file=sys.argv[2])
end = time.clock()
print "time: %f s" % (end - start)
'''
