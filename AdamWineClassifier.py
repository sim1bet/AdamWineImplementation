# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 08:42:01 2018

@author: smnbe
"""

from WineDataset import create_datasets
from AdamWineAlgo import to_one_hot, layers, lay_size, AdamModel, predict

X_Train, Y_Train, X_Test, Y_Test, m = create_datasets()
Y_Test = to_one_hot(Y_Test)
Y_Train = to_one_hot(Y_Train)
layers = layers()
lay_size = lay_size(X_Train, Y_Train, layers)
parameters = AdamModel(X_Train, Y_Train, lay_size, learning_rate=0.001, minibatch_size=64, beta1=0.9, beta2=0.999, epsilon=1e-8, n_epoch=1000, print_cost=True)
predict = predict(X_Test, Y_Test, parameters)