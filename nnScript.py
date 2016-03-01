import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt


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
    
    
    
def sigmoid(z):
    
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    
    return  1.0 / (1.0+np.exp(-z))
    
    

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
    # print len(mat.get('test1')[0])
    #Pick a reasonable size for validation data
    
    temp_train=[]
    temp_train_label=[]
    temp_test=[]
    test_label=[]



    for i in xrange(10):
        
        temp_train.extend(mat.get("train"+str(i)))
        for __ in xrange(len(mat.get("train"+str(i)))):
            temp_train_label.append([0 if j!=i else 1 for j in xrange(10)])
        
        temp_test.extend(mat.get("test"+str(i)))
        for __ in xrange(len(mat.get("test"+str(i)))):
            test_label.append([0 if j!=i else 1 for j in xrange(10)])


    temp_np_train=np.array(temp_train)
    test_data=np.array(temp_test)

    temp_np_train=temp_np_train/float(255)
    test_data=test_data/float(255)

    #TODO : Perform feature selection
    #Your code here


    permuted_train=np.random.permutation(temp_np_train)
    
    train_data = np.array(permuted_train[:50000])
    train_label = np.array(temp_train_label[:50000])
    validation_data = np.array(permuted_train[50000:])
    validation_label = np.array(temp_train_label[50000:])
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
    obj_val = 0
    sum_of_errors=0  
    #Your code here
    # for i,data in enumerate(training_data):
    i=0
    data=training_data[0]
    hidden_nodes_output = []
    data=np.append(data,1)              #Appending 1 as a dummy so that dot product can be calculated
    #Calculate the hidden nodes value matrix
    for input_weights in w1:
		sum_of_input_with_weights = sigmoid(np.dot(data, input_weights))
		hidden_nodes_output.append(sum_of_input_with_weights)
    hidden_nodes = np.append(hidden_nodes_output,1)      #Appending a 1 to hidden layer nodes

    #Calculate the output nodes value matrix
    output_nodes = []
    for hidden_weights in w2:
        sum_of_hidden_weights =sigmoid(np.dot(hidden_nodes,hidden_weights))
        output_nodes.append(sum_of_hidden_weights)

    #Equation 5
    sum_of_errors+=np.sum(np.square(np.subtract(output_nodes,training_label[i])))

    #Equation 6

    dabba=np.subtract(output_nodes,training_label[i])*(np.ones()-)
    hidden_nodes 
    total_error=sum_of_errors/len(training_data)


    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
    #obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.array([])
    
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

    labels = np.array([])
    for data_row in data:
        data_row = np.append(data_row, 1)
        hidden_nodes_output = np.array([])
        for input_weights in w1:
            sum_of_input_with_weights = sigmoid(np.dot(data_row, input_weights))        #Take the sigmoid of dot product of weights and input
            hidden_nodes_output = np.append(hidden_nodes_output, sum_of_input_with_weights)
        hidden_nodes_output = np.append(hidden_nodes_output, 1)
        for output_weights in w2:
            sum_of_output_weigthts = sigmoid(np.dot(output_weights, hidden_nodes_output))
            labels = np.append(labels.append, sum_of_output_weigthts) 
        labels.reshape(10,1)
        labels = labels.argmax(labels, 1)                                   #Return the index of number which has maximum value
    return labels
    



"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();


#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]; 

#print n_input

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 20;
				   
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

print('\n Test set Accuracy:' + + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
