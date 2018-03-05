# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 12:13:29 2018

@author: smnbe
"""

import numpy as np
import math

def create_datasets():
    dataset = np.genfromtxt('winequality-white.csv', delimiter=';')
    dataset = np.float32(dataset[1:,:])
    l = dataset.shape[0]
    perm = list(np.random.permutation(l))
    dataset = dataset[perm, :]
    X_Test=dataset[:64,:-1].T
    X_Train=dataset[64:,:-1].T
    m = X_Train.shape[1]
    Y_Test=dataset[:64,-1:]
    Y_Train=dataset[64:,-1:]
    Y_Train=Y_Train.astype(int)
    Y_Test=Y_Test.astype(int)
    Y_Train=np.squeeze(Y_Train)
    Y_Test=np.squeeze(Y_Test)
    
    return X_Train, Y_Train, X_Test, Y_Test, m


def shuffling(X_Train, Y_Train, m, minibatch_size):
    permutation = list(np.random.permutation(m))
    shuffled_X = X_Train[:, permutation]
    shuffled_Y = Y_Train[:, permutation]
    n_comp_minibatches = math.floor(m / minibatch_size)+1
    minibatches = [(shuffled_X[:,i*minibatch_size:(i+1)*minibatch_size],
                    shuffled_Y[:,i*minibatch_size:(i+1)*minibatch_size])
                   for i in range(n_comp_minibatches)]
    
    return minibatches
    

    




