import numpy as np
"""Functions for data preprossing
"""

def max_min_normalization(x):
    return ((x[np.newaxis,:,:]-np.mean(x,axis=0))/(np.max(x,axis=0)-np.min(x,axis=0)))[0]

def z_score_normalization(x):
    return ((x[np.newaxis,:,:] - np.mean(x,axis=0)) / np.std(x, axis = 0))[0]

def sigmoid_normalization(x):  
        return 1.0 / (1 + np.exp(-x));  

def standardize(x,mean=[],std=[]):
    if (len(mean) != 0) & (len(std) != 0):
        x_normalized = (x-mean)/std
    elif (len(mean) == 0) & (len(std) == 0):
        "mean of column, the same column means the same feature"
        mean = np.mean(x,axis=0)
        std = np.std(x,axis=0)
        x_normalized = (x-mean)/std
    else:
        raise NotImplementedError 
    return x_normalized,mean,std
