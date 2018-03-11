# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 08:42:01 2018

@author: smnbe
"""

from WineDataset import create_datasets
from AdamWineAlgo import to_one_hot, layers, lay_size, AdamModel, predict, accu

X_Train, Y_Train, X_Test, Y_Test, m = create_datasets()
print(Y_Test)
Y_Test = to_one_hot(Y_Test)
print(Y_Test)
Y_Train = to_one_hot(Y_Train)
layers = layers()
lay_size, lay_adam = lay_size(X_Train, Y_Train, layers)
parameters = AdamModel(X_Train, Y_Train, lay_size, lay_adam, learning_rate=0.0001, minibatch_size=64, beta1=0.9, beta2=0.999, epsilon=1e-8, n_epoch=1000, print_cost=True)
predict = predict(X_Test, Y_Test, parameters, epsilon=1e-8)
accuracy=accu(Y_Test, predict)
print ('Accuracy: %d' % (accuracy) + '%')