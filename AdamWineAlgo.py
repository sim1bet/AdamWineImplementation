# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 21:20:10 2018

@author: smnbe
"""

import numpy as np
import matplotlib.pyplot as plt
from WineDataset import shuffling

def to_one_hot(Y):
    nb_cl=11
    targets = Y.reshape(-1)
    Y = np.eye(nb_cl)[targets]
    Y=Y.T
    Y=Y.astype(int)
    
    return Y

def layers():
    n_lay=input("Define the number of hidden layers (max 4): ")
    while (not n_lay.isdigit()) or (int(n_lay)<=0 or int(n_lay)>=5):
        print("/nERROR! INPUT MUST BE AN INTEGER BETWEEN 1 AND 4")
        n_lay=input("Define the number of hidden layers (max 4): ")
    layers=int(n_lay)
    
    return layers

def lay_size(X_Train, Y_Train, layers):
    lay_size=[]
    units=X_Train.shape[0]
    lay_size.append(units)
    for l in range(layers):
        units=input("Define the n. of hidden units in layer "+str(l+1)+":")
        while (not units.isdigit()) or int(units)<=0:
            print("\nERROR! INPUT MUST BE AN INTEGER GREATER THAN 0")
            units=input("Define the n. of hidden units in layer "+str(l+1)+":")
        units=int(units)
        lay_size.append(units)
    units=Y_Train.shape[0]
    lay_size.append(units)
    lay_adam=np.copy(lay_size)
    
    return lay_size, lay_adam

def initialize_parameters(lay_size):
    parameters={}
    #Creates a dictionary containing weights W and biases b
    L=len(lay_size)
    for l in range(L-1):
        #He initialization for parameters Ws
        parameters["W"+str(l+1)]=np.random.randn(lay_size[l+1], lay_size[l])*0.01#(2/lay_size[l])
        parameters["b"+str(l+1)]=np.zeros((lay_size[l+1],1))
        
        assert(parameters["W"+str(l+1)].shape==(lay_size[l+1], lay_size[l]))
        assert(parameters["b"+str(l+1)].shape==((lay_size[l+1],1)))
    
    return parameters

def linear_forward(A, W, b):
    #Takes as arguments the activation of layer l-1,
    #the connection matrix W and the bias vector b of layer l
    Z=np.add(np.dot(W,A),b)
    
    linear_cache=(A, W, b)    
    #Stores the values A, W, b for each layer l--> useful for grad backprop
    
    return Z, linear_cache

def linear_act(Z, epsilon):
    #Computes non-linear activation for each layer l of units
    #--> ReLu function : {A=0 if Z<=0
    #                     A=Z if Z>0}
    A_rl=np.maximum(epsilon, Z)
    activation_cache=Z
    #Cache storing linear activation of layer l (for backprop)
    
    assert(A_rl.shape==Z.shape)
    
    return A_rl, activation_cache

def linear_softmax(Z):
    #Computes the softmax activation for vector A
    #Activation values within [0,1]
    #shiftA shifts values for A_rl closer to zero so that, once exponentiated
    #they don't explode (all negative except one value - many approximated to zero)
    A=np.exp(Z-np.max(Z))/np.sum(np.exp(Z-np.max(Z)), axis=0)
    activation_cache=Z
    #Cache storing ReLu activation of layer l (for backprop)
    
    assert(A.shape==Z.shape)
    
    return A, activation_cache

def linear_act_forward(A_prev, W, b, epsilon, act):
    #Takes as argument the activation A_prev of layer l-1,
    #the connection matrix W and the bias vector b of layer l
    Z, linear_cache = linear_forward(A_prev, W, b)
    if act=="ReLu":
        A, activation_cache = linear_act(Z, epsilon)
    elif act=="Softmax":
        A, activation_cache = linear_softmax(Z)
    cache=(linear_cache, activation_cache)
    
    return A, cache

def L_lay_forw(X, parameters, epsilon):
    #Iterates the linear_act_forward process across the entire architecture
    #stores all the "cache" in a caches list
    #Stores all the activations in a A_l list
    caches=[]
    L=len(parameters)//2
    A=X
    for l in range(1,L+1):
        A_prev=A
        if l!=L:
            act="ReLu"
        elif l==L:
            act="Softmax"
        A, cache = linear_act_forward(A_prev, parameters["W"+str(l)], parameters["b"+str(l)], epsilon, act)
        caches.append(cache)
    AL= np.copy(A)
    
    assert(AL.shape==X.shape)
    
    return AL, caches, c

def L2_comp(X, parameters, lambdal2):
    m=X.shape[1]
    L2=0
    L=len(parameters)//2
    for l in range(L-1):
        L2reg=(1/m)*(lambdal2/2)*(np.sum(np.square(parameters["W"+str(l+1)])))
        L2+=L2reg
        
    return L2

def compute_cost(AL, Y_Train, L2):
    m=Y_Train.shape[1]
    logprobs=np.multiply(np.abs(Y_Train),np.log(np.abs(AL)))+np.multiply((1-np.abs(Y_Train)),np.log(1-np.abs(AL)))
    xent=-np.sum(logprobs)/m+L2
    
    cost=np.squeeze(xent)  #Ensures that there are no irrelevant dimensions casted in
    
    return cost

def Soft_back(Y_Train, dAL):
    #Computes the gradient with respect to softmax activation
    #Only for the last layer
    dZ=np.multiply(Y_Train, dAL)
    s = dZ.sum(axis=dZ.ndim-1, keepdims=True)
    dZ += np.multiply(Y_Train, s) 
    
    return dZ

def rl_back(activation_cache):
    #Computes gradient with respect to softmax activation
    #For all layers except the last
    Z=activation_cache
    dZ=np.copy(Z)
    dZ[Z<=0]=0
    dZ[Z>0]=1
    #if Z<=0 --> dZ=0
    #if Z>0 --> dZ=1
    
    assert(dZ.shape==Z.shape)
    
    return dZ

def linear_back(dZ, linear_cache):
    A_prev, W, b = linear_cache
    m=A_prev.shape[1]
    #Retrieve some necessary values
    dW=np.dot(dZ, A_prev.T)/m
    db=np.sum(dZ, axis=1, keepdims=True)/m
    dA_prev=np.dot(W.T, dZ)
    
    assert(dW.shape==W.shape)
    assert(db.shape==b.shape)
    assert(dA_prev.shape==A_prev.shape)
    
    return dA_prev, dW, db

def linear_activation_back(Y_Train, dA_prev, cache, act):
    #Divides the backward process according to 
    #Softmax or ReLu activation and recall cache from caches[]
    linear_cache, activation_cache = cache
    
    if act=="Softmax":
        dZ=Soft_back(Y_Train,dA_prev)
        dA_prev, dW, db = linear_back(dZ, linear_cache)
    elif act=="ReLu":
        dZ=rl_back(activation_cache)
        dA_prev, dW, db = linear_back(dZ, linear_cache)
    
    return dA_prev, dW, db

def L_lay_back(Y_Train, AL, caches, parameters):
    #Creates a dictionary of parameters to later use them for Adam
    grads={}
    L=len(caches)
    m=Y_Train.shape[1]
    lambd=2.3e-3
    dAL=AL-Y_Train
    #Through the definition of act, we implement a unique loop for the backprop
    for l in reversed(range(L)):
        if l==L-1:
            act="Softmax"
            grads["dA"+str(l+1)]=dAL
        elif l!=L-1:
            act="ReLu"
        dA_prev_temp, dW_temp, db_temp = linear_activation_back(Y_Train, grads["dA"+str(l+1)], caches[l], act)
        grads["dA"+str(l)]=dA_prev_temp
        grads["dW"+str(l+1)]=dW_temp+(1/m)*parameters["W"+str(l+1)]*lambd
        grads["db"+str(l+1)]=db_temp
        
    return grads

def initialize_Adam(lay_adam):
    #Takes parameters as arguments for size
    #Initialize velocity (momentum) and exponentially weighted average of the squared gradient
    L=len(lay_adam)
    v={}
    s={}
    #Initialization of v and s
    for l in range(L-1):
        v["dW"+str(l+1)]=np.zeros((lay_adam[l+1], lay_adam[l]))
        s["dW"+str(l+1)]=np.zeros((lay_adam[l+1], lay_adam[l]))
        v["db"+str(l+1)]=np.zeros((lay_adam[l+1],1))
        s["db"+str(l+1)]=np.zeros((lay_adam[l+1],1))
        
    return v, s

def upd_para_adam(parameters, grads, v, s, t, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
    L=len(parameters)//2
    v_corrected={}
    s_corrected={}
    #Encapsulate the updated values for v and s, which are dependent on grads
    #Performance of Adam update on all parameters
    for l in range(L):
        #Moving average of the gradients with momentum
        v["dW"+str(l+1)]=np.multiply(beta1,v["dW"+str(l+1)])+np.multiply((1-beta1),grads["dW"+str(l+1)])
        v["db"+str(l+1)]=np.multiply(beta1,v["db"+str(l+1)])+np.multiply((1-beta1),grads["db"+str(l+1)])
        
        #Coreccting biases for v values across layers
        v_corrected["dW"+str(l+1)]=v["dW"+str(l+1)]/(1-np.power(beta1, t))
        v_corrected["db"+str(l+1)]=v["db"+str(l+1)]/(1-np.power(beta1, t))
        
        #Moving averages according to square gradients
        s["dW"+str(l+1)]=np.multiply(beta2,s["dW"+str(l+1)])+np.multiply((1-beta2),np.power(grads["dW"+str(l+1)],2))
        s["db"+str(l+1)]=np.multiply(beta2,s["db"+str(l+1)])+np.multiply((1-beta2),np.power(grads["db"+str(l+1)],2))
        
        #Correcting biases for s values across layers
        s_corrected["dW"+str(l+1)]=s["dW"+str(l+1)]/(1-np.power(beta2, t))
        s_corrected["db"+str(l+1)]=s["db"+str(l+1)]/(1-np.power(beta2, t))
        
        #Updating parameters procedure
        parameters["W"+str(l+1)]=np.add(parameters["W"+str(l+1)],np.multiply((-learning_rate),(v_corrected["dW"+str(l+1)]/np.sqrt(s_corrected["dW"+str(l+1)]+epsilon))))
        parameters["b"+str(l+1)]=np.add(parameters["b"+str(l+1)],np.multiply((-learning_rate),(v_corrected["db"+str(l+1)]/np.sqrt(s_corrected["db"+str(l+1)]+epsilon))))
        
    return parameters, v, s

def AdamModel(X_Train, Y_Train, lay_size, lay_adam, learning_rate, minibatch_size, beta1, beta2, epsilon, n_epoch, print_cost):
    #Implements the complete model
    #Incudes shuffling of minibatches at each epoch
    costs=[]
    t=0             #Initialize the counter for Adam update +1 at each epoch
    m=X_Train.shape[1]
    
    #Initialization of parameters
    parameters = initialize_parameters(lay_size)
    #L=len(lay_size)
    
    #Initialization of v, s for Adam
    v, s = initialize_Adam(lay_adam)
    
    #iterates the procedure for n_epoch
    for n in range(n_epoch):
        #Permutation of X_Train, Y_Train and creation of minibatches
        minibatches = shuffling(X_Train, Y_Train, m, minibatch_size)
        
        #Iterate the forward-backward procedure for each minibatch
        for minibatch in minibatches:
            
            #Unpacking of minibatch content
            minibatch_X, minibatch_Y = minibatch
            
            #Forward-prop for the minibatch
            AL, caches = L_lay_forw(minibatch_X, parameters, epsilon)
            
            #Computes the regularizing term L2
            L2=L2_comp(minibatch_X, parameters, lambdal2=2.3e-3)
            
            #Computes the cost associated to the output of the minibatch
            cost = compute_cost(AL, minibatch_Y, L2)
            
            #Computation of gradients
            grads = L_lay_back(minibatch_Y, AL, caches, parameters)
            
            #Parameters updating procedure
            t +=1
            parameters, v, s = upd_para_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon)
            
        if print_cost and n%20==0:
            print ("Cost after epoch %i: %f" %(n, cost))
        if print_cost and n%20==0:
            costs.append(cost)
            
    #synaptic pruning
    #for l in range(L-1):
     #   parameters["W"+str(l+1)][np.where(parameters["W"+str(l+1)]<0.05)]=0
     #   parameters["b"+str(l+1)][np.where(parameters["b"+str(l+1)]<0.05)]=0
        
    #Plot the graph related to the learning instance
    plt.plot(costs)
    plt.ylabel("Cost")
    plt.xlabel("Epoch (per twentieth)")
    plt.title("Learning rate :"+str(learning_rate))
    plt.show()
    
    return parameters

def predict(X_Test, Y_Test, parameters, epsilon):
        #Functions that predicts value for X_Test
        #Computation of final activation for X_Test
        c=0
        AL, caches, c = L_lay_forw(X_Test, parameters, epsilon, c)
        
        #Creation of the prediction matrix
        predict=np.copy(AL)
        
        #Deterministic activation
        predict[np.where(predict>0.15)]=1
        predict[np.where(predict<=0.15)]=0
        
        return predict
    
def accu(Y_Test,predict):
    accuracy=np.multiply((Y_Test),(predict))
    accuracy=(np.sum(accuracy)/float(Y_Test.shape[1])*100)
    
    return accuracy
    
    
        
        
        
        
        
        
        
        
    
    
    
    
    
    
