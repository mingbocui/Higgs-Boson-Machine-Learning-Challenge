# coding: utf-8

import numpy as np

from proj1_helpers import load_csv_data, create_csv_submission
from feature_selection import select_best_degrees, build_poly_by_feature
from feature_selection import compute_log, compute_theta, compute_physics
from correction_rate import cross_validation, print_score
from evaluation import predict_regression_labels
from costs import sigmoid


# edit if train.csv and test.csv in not in ../data/
dat_dir = '../data/'

############################
## Training
############################
print('training started')

# load the training set
print('loading the training dataset...')
y_train_pre, tx_train, ids_train = load_csv_data(dat_dir + "train.csv", sub_sample=False)
print('data loaded...')

y_train=y_train_pre.reshape(y_train_pre.shape[0],1)

# construct the featues using log()
index_log = [0,1,2,4,5,6,7,9,10,12,16,21,23,24,25,26,27,28,29]
tx_log,mean_log,std_log = compute_log(tx_train, index_log)

# construct the featues using cosine()
index_theta = [14,15,17,18,20]
tx_theta,mean_theta,std_theta = compute_theta(tx_train,index_theta)

# construct the featues with physics meanings: 
# index_physics_A (mass) * index_physics_B / index_physics_C
index_physics_A = [0]
index_physics_B = [10,13,13,9]
index_physics_C = [9,16,21,10]
tx_physics, mean_physics, std_physics = compute_physics(tx_train,
                                                        index_physics_A,
                                                        index_physics_B,
                                                        index_physics_C)

# combine all the selected features for training set
train_new = np.c_[tx_log, tx_theta, tx_physics]


# calculate the best degrees for the features
# # using the least_squares regression
'''unindented the following codes for debugging'''
# best_degrees = select_best_degrees(y_train, train_new, max_degree=4)
# np.save('best_degrees.npy',best_degrees)
best_degrees = np.load('best_degrees.npy')

# reconstruct all the features at their own best degrees
train_best_degrees = build_poly_by_feature(train_new, best_degrees)

# innitialize hyper parameters
k_fold = 15
method = 'RR'
lambda_ = 1e-7

# Normalization
X_sigmoid = sigmoid(train_best_degrees)

y = np.copy(y_train)

initial_w_pre = np.zeros((np.size(X_sigmoid,1)))
initial_w=initial_w_pre.reshape(initial_w_pre.shape[0],1)

# ridge regression
scores, weights, loss_avg, score_avg = cross_validation(y, X_sigmoid, 'RR', {"lambda_": lambda_}, 
                                                        threshold = 0, k=k_fold, seed=0)
print_score(loss_avg, score_avg, scores, method='Ridge regression')

# select the best weights
'''unindented the following codes for debugging'''
# best_weights = weights[np.where(scores == np.max(scores))[0][0]]
# np.save('best_weights.npy',best_weights)
best_weights = np.load('best_weights.npy')

print('training finished')

############################
## Prediction
############################
print('prediction started')

# load the test set
print('loading the testing dataset...')
y_test, tx_test,ids_test = load_csv_data(dat_dir +"test.csv")
print('data loaded...')

# combine all the selected features for testing set
# Note we used the same means and stds from training set
test_log,_,_ = compute_log(tx_test, index_log, mean_log, std_log)
test_theta,_,_ = compute_theta(tx_test, index_theta, mean_theta, std_theta)
test_physics, _, _ = compute_physics(tx_test, index_physics_A, index_physics_B, index_physics_C,
                                     mean_physics, std_physics)
test_new = np.c_[test_log,test_theta,test_physics]


# reconstruct all the features of test set using the best degrees from training set
test_best_degree = build_poly_by_feature(test_new, best_degrees)
# Normalization
X_test = sigmoid(test_best_degree)

# predict
y_pred = predict_regression_labels(best_weights, 
                                   X_test, threshold=0)

print('prediction ended')

# generate submission
create_csv_submission(ids_test, y_pred, 'submission.csv')
print('submission generated')