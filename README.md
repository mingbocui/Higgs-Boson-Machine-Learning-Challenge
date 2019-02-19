# Project 1 for course CS-433 Machine Learning 

	.
	|-- README.md
	|-- data (Please put train.csv and test.csv here)
	|-- src
	|   -- correction_rate.py
	|   -- costs.py
	|   -- evaluation.py
	|   -- feature_selection.py
	|   -- gradient_optimization.py
	|   -- implementations.py
	|   -- preprocess.py
	|   -- proj1_helpers.py
	|   -- run.py
	|   -- run.ipynb
    |   -- test.ipynb
	|   -- exploring data analysis.ipynb

## Introduction:
In Project 1, machine learning methods were applied on the CERN dataset.
To make sure the codes can work properly, please put the 'train.csv' and 'test.csv' files under the folder 'data'.

## Process of buliding models and getting prediction results:
In src folder is presented the required run.py file that implements the training of our best model and three notebooks that give examples of how we optimized the various methods.

The run.py file implements how we train our best model and also generates the results in the file 'submission.csv'. 
The notebook run.ipynb shows extra information in addition to run.py. 
The notebook test.ipynb shows how we test and tune all the models from implementation.py using cross validation. 
The notebook exploring data analysis.ipynb shows how we do exploratory data analysis.

1. run.py:
In this file, we have two parts, which are training part and testing part. In training part, we do the feature engineering such as constructing the featues with physics meanings and selecting the best degrees for the features. Then, we build the Ridge Regression model. In the testing part, we load the dataset and use our model to predict the results. Finally, we export the results into 'submission.csv' file.

2. run.ipynb:
In the notebook, we have three parts. The first two parts are the same as the training part and testing part in run.py. The third part is "other methods", which shows the performance of other models. We print out the loss and accuracy for the models. 

3. exploring data analysis.ipynb:
In this notebook, we do the data analysis. For example, we calculate the correlation coefficient between different features. Then, we use some vitualization methods to show our findings.

## Functions implementations:
We implemented many functions to build our model. Here is the brief description for the functions:

1. 'implementations.py':
    Note that aparting from the required input, we add an extra variable called 'loss_method' which enable users to test 'mse', 'mae' and 'rmse' when computing loss and gradient. Genrally, 'loss_method' is 'mse' by default.

   1. least_squares(y, tx, loss_method='mse'): calculate the least squares solution.
   2. ridge_regression(y, tx, lambda_, loss_method='mse'): implement ridge regression. Minimization of the penalized mean squared error with the ridge regularization.
   3. least_squares_GD(y, tx, initial_w, max_iters, gamma, method='mse'): Linear regression using gradient descent.
   4. least_squares_SGD(y, tx, initial_w, max_iters, gamma, batch_size=1, method='mse'): Linear regression using stochastic gradient descent.
   5. logistic_regression(y, tx, initial_w, max_iters, gamma)
   6. reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)
   7. print_step(n_iter, step_wise, max_iters, loss, w, method = 'unknown')
   8. print_head()
	
2. 'costs.py':
    We unify the methods to calculate gradient and loss here
   1. sigmoid(t): apply sigmoid function on t.
   2. calculate_sigmoid(y, tx, w): compute the cost by negative log likelihood.
   3. calculate_mse(err): Calculate the mse for vector err.
   4. calculate_mae(err): Calculate the mae for vector err.
   5. compute_loss(y, tx, w, loss_method='mse'): Calculate the loss using mse, mae, rmse or sigmoid.
   6. def compute_gradient(y, tx, w, loss_method='mse'): Compute the gradient of the cost function for linear regression you can use mse or mae

3. 'evaluation.py':
    This script is at present only using accuracy as the model selection criterion.

   1. predict_regression_labels(weights, data, threshold=0): Generates regression label given weights, and a test data matrix.
   2. predict_logistic_labels(weights, data, threshold=0.5): Generates logistic classification given weights, and a test data matrix.
   3. regresssion_score(y, tx, w, threshold=0): calculates the final score of a given method.
   4. logistic_score(y, tx, w, threshold=0.5): Generates class predictions given weights, and a test data matrix for logistic regression methods
   
4. 'feature_selection.py':
    We stored the methods to process features here.

   1. test_accuracy(y, tx, w): calculates the final score of a given method
   2. compute_log(tx, index_log, mean=[], std=[]): compute the log value of the given features, and then standardize the features and return mean and std
   3. compute_theta(tx, index_theta, mean=[],std=[]): compute the cosine value of the angle-like features, and then standardize the features and return mean and std
   4. compute_physics(tx, index_A, index_B, index_C, mean=[], std=[]): add the features that have physics means, using the particle mass (index_A = 0) as the weight, and then standardize the features and return mean and std
   5. select_best_degrees(y, tx, reg_function=least_squares, max_degree=5): Compute the optimal polynomial expansion degree for each feature individually.
   6. build_poly_by_feature(tx, degrees): Builds a polynomial expansion based on the given degree for each feature.
   
5. 'gradient_optimization.py':
    To make implementation,py look better, we keep the optimization loops here.

   1. update_gd(y, tx, w, ws, losses, gamma, lambda_=0, method='mse')
   2. update_sgd(y, tx, w, ws, losses, gamma, lambda_=0, batch_size=1, method='mse')

6. 'correction_rate.py': 
    Here we developed the cross validation functions.

   1. build_k_indices(y, k_fold, seed): build k indices for k-fold.
   2. run_method(y, tx, method, params): Run the wanted method and returns its value.
   3. cross_validation(y, tx, method, params,threshold=0, k=5, seed=0): Gives the mean RMSE for running the given method with the given parameters.
   4. test_LS(y, tx)
   5. test_GD(y, tx, gamma)
   6. test_SGD(y, tx, gamma)
   7. test_LR(y, tx, gamma)
   8. test_RLR(y, tx, lambda_, gamma)
   9. print_score(loss_avg, score_avg, scores, method='Ridge regression')

7. 'preprocess.py':
   This is for standarzation and normalization.
   1. max_min_normalization(x)
   2. z_score_normalization(x)
   3. sigmoid_normalization(x)
   4. standardize(x,mean=[],std=[])
