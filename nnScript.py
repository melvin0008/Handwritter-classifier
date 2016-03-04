import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import random
import time


def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W
    
def featureReduction(data):
    deleteIndices = [];
    #Tweaks added for optimizing
    for i in range(0,data.shape[1]):
        if ((data[:,i] - data[0,i]) == 0).all():
            deleteIndices += [i];
    #data_temp = np.delete(data,deleteIndices,1)
    return deleteIndices
    
def get_dummies(label):
    rows = label.shape[0];
    rowsIndex=np.arange(rows,dtype="int")
    # Below line can be hardcoded in our case 
    oneKLabel = np.zeros((rows,10))
    #oneKLabel = np.zeros((rows,np.max(label)+1))
    oneKLabel[rowsIndex,label.astype(int)]=1
    return oneKLabel

def sigmoid(z):
    
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    
    return  1 / (1+np.exp(np.multiply(-1,z)))
    
    

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""
    
    mat = loadmat('mnist_all.mat') #loads the MAT object as a Dictionary
    
    temp_train_label=[]
    temp_test=[]
    temp_train = []
    test_label = []

    for i in xrange(10):
        temp_train.extend(mat.get("train"+str(i)))
        for __ in xrange(len(mat.get("train"+str(i)))):
            temp_train_label.append(i)
        
        temp_test.extend(mat.get("test"+str(i)))
        for __ in xrange(len(mat.get("test"+str(i)))):
            test_label.append(i)

    temp_np_train=np.array(temp_train)
    test_data=np.array(temp_test)
    
    
    temp_np_train=np.double(temp_np_train)
    test_data=np.double(test_data)
    temp_train_label=np.double(temp_train_label)
    test_label=np.double(test_label)

    temp_np_train=temp_np_train/255.0
    test_data=test_data/255.0

    #TODO : Perform feature selection
    #Your code here
    deleteIndices = featureReduction(temp_np_train)
    
    # # Get Reduced train and test
    temp_np_train = np.delete(temp_np_train,deleteIndices,1)
    test_data = np.delete(test_data, deleteIndices,1)
    
    array_permutation = np.random.permutation(xrange(len(temp_np_train)))
    
    train_data = temp_np_train[array_permutation[:50000],:]
    train_label = temp_train_label[array_permutation[:50000]]; 
    validation_data =temp_np_train[array_permutation[50000:],:]
    validation_label = temp_train_label[array_permutation[50000:]]; 
    
    print 'Train_data='+str(train_data.shape)
    print 'train_label='+str(train_label.shape)
    print 'valiation_data='+str(validation_data.shape)
    print 'Validation_label='+str(validation_label.shape)
    print 'Test_data='+str(test_data.shape)
    print 'test_label='+str(test_label.shape)
    
    return train_data, train_label, validation_data, validation_label, test_data, test_label
    
    
def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""
    
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    #print 'W1_shape='+str(w1.shape)
    #print 'W2_shape='+str(w2.shape)
    trans_w1=w1.T
    trans_w2=w2.T
    # print training_label[0]
    # print training_label.shape
    training_label = get_dummies(np.array(training_label))
    # print training_label[0]
    # print training_label.shape
    #add bias 1
    x=np.column_stack((training_data,np.ones(len(training_data))))
    #equation1
    eq1=np.dot(x,trans_w1)
    #equation 2
    z=sigmoid(eq1)
    #add bias 1
    z=np.column_stack((z,np.ones(len(z))))
    #equation 3
    eq3=np.dot(z,trans_w2)
    #equation 4
    o=sigmoid(eq3)
    
    delta=np.subtract(o,training_label)
    eq5=np.sum(np.square(delta))

    dabba=(training_label-o)*(1-o)*o
    
    grad_w2=np.multiply(-1,np.dot(dabba.T,z))
    

    one_minus_z_into_z = (1-z)*z
    
    
    multiply_by_summation = one_minus_z_into_z*np.dot(dabba,w2)
    
    grad_w1_without_minus_one = np.dot(np.transpose(multiply_by_summation),x)
    

    grad_w1=np.multiply(-1,grad_w1_without_minus_one)
    
    grad_w1 = np.delete(grad_w1, n_hidden,0)
    
    obj_grad = np.array([])
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad=obj_grad/len(training_data)
    
    obj_val=eq5/len(training_data)
    
    return (obj_val,obj_grad)


def nnPredict(w1,w2,data):
    
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels""" 
    
    #Your code here
    #This function is similar to the initial calculation in nnObjFunction
    trans_w1=w1.T
    trans_w2=w2.T

    x=np.column_stack((data,np.ones(len(data))))
    #equation1
    eq1=np.dot(x,trans_w1)
    
    #equation2
    z=sigmoid(eq1)

    z=np.column_stack((z,np.ones(len(z))))

    eq3=np.dot(z,trans_w2)

    o=sigmoid(eq3)

    labels = np.argmax(o, 1)                                   #Return the index of number which has maximum value
    return labels
    



"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();


#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]; 



# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 1;
				   
# set the number of nodes in output unit
n_class = 10;				   

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 0;


args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))


#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)


#find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')


predicted_label = nnPredict(w1,w2,test_data)

#find the accuracy on Validation Dataset
print('\n Test set Accuracy:' + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')