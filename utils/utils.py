import numpy as np 
import numpy_indexed as npi

def mse(ytrue,ypred):
    """ Mean squared error cost function """

    N = len(ytrue)
    return np.sum((ytrue-ypred)**2)/N


def cross_entropy(ytrue,ypred):
    """ Binary cross entropy cost function """

    N = len(ytrue)
    return -np.sum(ytrue*np.log(ypred)+(1-ytrue)*np.log(1-ypred))/N

def unique_values(X, feat, decimal):
    """ Keep only values of a given feature, feat, thereby reducing the dataset X """

    _, Xred = npi.group_by(np.array(X)[:,feat].round(decimals=decimal)).min(np.array(X))
    return Xred

def binarize_data(X,features):
    """ Transform all features into binary  """

    for f in range(X.shape[1]):
        mean = round(np.mean(X[:,f]), 3) 
        X[:,f] = (X[:,f] >= mean)
        features[f] += " >= " + str(mean)

    return features, X

def print_instance(X,features, num):
    """ Print all values for a given point """

    print("Instance number : ", num)

    for i in range(len(features)):
        print(features[i], " = ", X[i] )
    





