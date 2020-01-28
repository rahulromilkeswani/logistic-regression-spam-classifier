import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt

#copied from the discussion link
def sigmoid(x):
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        # if x is less than zero then z will be small, denom can't be
        # zero because it's 1+z.
        z = np.exp(x)
        return z / (1 + z)


def predict(X, w):
    n_ts = X.shape[0]
    # use w for prediction
    pred = np.zeros(n_ts)       # initialize prediction vector
    for i in range(n_ts):
        y_value = np.dot(X[i],np.transpose(w))   #taking dot product to find the y-value. 
        #predicting label based on y-value
        if( y_value> 0) :
            pred[i] = 1 
        else : 
            pred[i] = 0
    return pred


def accuracy(X, y, w):
    y_pred = predict(X, w)
    accuracy_rate = ((y==y_pred).mean())*100  #checking for accuracy between predicted and actual label
    return accuracy_rate

def logistic_reg(X_tr, X_ts, lr):
    iteration_list=[]
    train_accuracy_list = []
    test_accuracy_list=[]
    #perform gradient descent
    n_vars = X_tr.shape[1]  # number of variables
    n_tr = X_tr.shape[0]    # number of training examples
    w = np.zeros(n_vars)    # initialize parameter w
    tolerance = 0.01        # tolerance for stopping

    iter = 0                # iteration counter
    max_iter = 1000         # maximum iteration

    while (True):
        iter += 1

        # calculate gradient
        grad = np.zeros(n_vars) # initialize gradient
        for i in range(n_tr):
            temp = np.dot(X_tr[i],np.transpose(w))
            temp_sigmoid = sigmoid(temp)
            for j in range(n_vars):
                grad[j] = grad[j] + (y_tr[i] - temp_sigmoid) * X_tr[i][j]  #computing gradient

        w_new = w + lr * grad  #updating coeffient values

        #printing gradient after 50 iterations and also calculating efficiency of train and test data. 
        if(iter%50 == 0):
            print('Iteration: {0}, mean abs gradient = {1}'.format(iter, np.mean(np.abs(grad))))
            test_accuracy  = accuracy(X_ts, y_ts, w_new)
            train_accuracy = accuracy(X_tr, y_tr, w_new)
            train_accuracy_list.append(train_accuracy)
            test_accuracy_list.append(test_accuracy)
            iteration_list.append(iter)
            print(train_accuracy)
            print(test_accuracy)



        # stopping criteria and perform update if not stopping
        if(np.mean(np.abs(grad)) < tolerance):
            w = w_new
            break
        else :
            w = w_new 

        if (iter >= max_iter):
            break

    #test_accuracy  = accuracy(X_ts, y_ts, w)
    #train_accuracy = accuracy(X_tr, y_tr, w)
    
    return test_accuracy, train_accuracy, iteration_list, train_accuracy_list, test_accuracy_list

def logistic_reg_regularized(X_tr, X_ts, k):
    lr = 1e-3
    #perform gradient descent
    n_vars = X_tr.shape[1]  # number of variables
    n_tr = X_tr.shape[0]    # number of training examples
    w = np.zeros(n_vars)    # initialize parameter w
    tolerance = 0.01        # tolerance for stopping
    iter = 0                # iteration counter
    max_iter = 1000         # maximum iteration
    penalty = (2**k)
    while (True):
        iter += 1
        # calculate gradient
        grad = np.zeros(n_vars) # initialize gradient
        for i in range(n_tr):
            temp = np.dot(X_tr[i],np.transpose(w))
            temp = sigmoid(temp)
            for j in range(n_vars):
                grad[j] = grad[j] + ((y_tr[i] - temp) * X_tr[i][j])  #calculating gradient with a penalty term
        w_new = w + (lr * (grad - (penalty*w))) #updating coefficient values 
        # stopping criteria and perform update if not stopping
        if(np.mean(np.abs(grad)) < tolerance):
            w = w_new
            break
        else :
            w = w_new 

        if (iter >= max_iter):
            break
    #computing accuracy
    test_accuracy  = accuracy(X_ts, y_ts, w)  
    train_accuracy = accuracy(X_tr, y_tr, w)
  
    return test_accuracy, train_accuracy

# read files
D_tr = genfromtxt('spambasetrain.csv', delimiter = ',', encoding = 'utf-8')
D_ts = genfromtxt('spambasetest.csv', delimiter = ',', encoding = 'utf-8')

# construct x and y for training and testing
X_tr = D_tr[: ,: -1]
y_tr = D_tr[: , -1]
X_ts = D_ts[: ,: -1]
y_ts = D_ts[: , -1]

# number of training / testing samples
n_tr = D_tr.shape[0]
n_ts = D_ts.shape[0]

# add 1 as feature
X_tr = np.concatenate((np.ones((n_tr, 1)), X_tr), axis = 1)
X_ts = np.concatenate((np.ones((n_ts, 1)), X_ts), axis = 1)

# learning rate values
learning_rates = [1e-3]
#learning_rates = [1,0.01,0.0001,0.000001]

for rate in learning_rates : 
    print('LEARNING RATE = ' + str(rate))
    test_accuracy, train_accuracy, iteration_list, train_accuracy_list, test_accuracy_list = logistic_reg(X_tr, X_ts,rate)
    #plotting graph
    plt.plot(iteration_list, train_accuracy_list , label = 'Train Accuracy')
    plt.plot(iteration_list,test_accuracy_list,  label = 'Test Accuracy')
    plt.title('Learning Rate = ' + str(rate))
    plt.legend()
    plt.show()
    print(train_accuracy_list)
    print(test_accuracy_list)
    print('learning rate =  {0}, train accuracy = {1}, test_accuracy = {2}'.format(str(rate),str(train_accuracy), str(test_accuracy)))

#k-values
k_values = [-8,-6,-4,-2,0,2]
train_accuracies = []
test_accuracies = []
for k_value in k_values : 
    test_accuracy, train_accuracy = logistic_reg_regularized(X_tr,X_ts,k_value)
    print(test_accuracy)
    print(train_accuracy)
    print()
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

#plotting graph
plt.plot(k_values, train_accuracies, label = 'Training Accuracy')
plt.plot(k_values, test_accuracies, label = 'Testing Accuracy')
plt.title('K-VALUE vs Train/Test Accuracy')
plt.legend()
plt.show()
