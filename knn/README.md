@ introduction:
knn realization on python

@ file introduction:
'knn.py' --- basic linear searching
'knn_kd.py' --- use kd_tree to search k-nearest neighbor
'knn_kd_scratch' --- use kd_tree to search k-nearest neighbor, and kd_tree is realized from scratch, see 'kdtree.py'
'kdtree.py' --- a kdtree, including kdtree's construction and k-nearest-seach

@example USAGE:
python knn.py 10 train.csv test.csv   
# 10 indicates the knn's k, train.csv is the trainning samples, test.csv is the testing samples

python knn_kd.py 10 train.csv test.csv
# the same of knn.py

python knn_kd_scratch.py 10 train.csv test.csv
# the same of knn.py