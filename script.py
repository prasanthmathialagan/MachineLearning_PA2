import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix


    # IMPLEMENT THIS METHOD

    d = X.shape[1]                  # Number of attributes in the input
    y = y.reshape(y.size)           # Flattening the output matrix containing class labels

    diffClasses = np.unique(y)      # Getting the matrix containing only the unique labels
    k = np.prod(diffClasses.shape)  # Getting the count of unique labels

    means = np.zeros((d, k))        # initializing the means matrix with 0's and size d x k where k is number of unique output labels

    for cl in range(diffClasses.size):
        means[:,cl] = np.mean(X[y == diffClasses[cl]],0)    #Calculating the mean corresponding to different classes

    covmat = np.cov(X.T)            #Covariance matrix for the pooled data

    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes

    # IMPLEMENT THIS METHOD

    d = X.shape[1]                                      # Number of attributes in the input
    y = y.reshape(y.size)                               # Flattening the output matrix containing class labels

    diffClasses = np.unique(y)                          # Getting the matrix containing only the unique labels
    k = np.prod(diffClasses.shape)                      # Getting the count of unique labels

    means = np.zeros((d,k))                             # initializing the means matrix with 0's and size d x k where k is number of unique output labels
    covmats = []                                        # initializing the list containing covariance matrices for each of the k classes

    for cl in range(diffClasses.size):
        means[:, cl] = np.mean(X[y == diffClasses[cl]], 0)
        covmats.append(np.cov(X[y == diffClasses[cl]].T))               #Calculating the mean and covariance for each of the classes seperately

    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    N = Xtest.shape[0]                  # Calculate the number of input samples from the given data
    d = Xtest.shape[1]                  # Number of attributes in the input
    nClasses = means.shape[1]           # Getting the count of unique labels/classes

    Pmat = np.zeros((N, nClasses))      # Probability matrix for classifying the input to different labels


    for xin in range(N):
        for m in range(nClasses):
            term = -(np.dot((Xtest[xin]-means[:, m].T).T, np.dot(inv(covmat), (Xtest[xin]-means[:, m].T))))/2.0
            expTerm = np.exp(term)
            den = pow((2*pi), d/2.0)*sqrt(det(covmat))
            Pmat[xin,m] = expTerm/den                      # Equation for Normal distribution

    maxIndices = np.argmax(Pmat, axis=1)+1                # Getting the indices of the column containing the max probability for each of the rows
    ypred = np.array(maxIndices)[np.newaxis].T            # Converting it to a column vector

    matchCount = 0
    for i in range(N):
        if ypred[i] == ytest[i]:                           # Getting the count of matching labels b/w predicted and true labels
            matchCount += 1

    acc = matchCount/N * 100                               # Calculating the accuracy manually

    return acc,ypred


def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD

    N = Xtest.shape[0]                  # Calculate the number of input samples from the given data
    d = Xtest.shape[1]                  # Number of attributes in the input
    nClasses = means.shape[1]           # Getting the count of unique labels/classes

    Pmat = np.zeros((N, nClasses))      # Probability matrix for classifying the input to different labels

    for xin in range(N):
        for m in range(nClasses):
            term = -(np.dot((Xtest[xin] - means[:, m].T).T, np.dot(inv(covmats[m]), (Xtest[xin] - means[:, m].T)))) / 2.0
            expTerm = np.exp(term)
            den = pow((2 * pi), d / 2.0) * sqrt(det(covmats[m]))
            Pmat[xin, m] = expTerm / den    # Equation for Normal distribution

    maxIndices = np.argmax(Pmat, axis=1) + 1                # Getting the indices of the column containing the max probability for each of the rows
    ypred = np.array(maxIndices)[np.newaxis].T              # Converting it to a column vector

    matchCount = 0
    for i in range(N):
        if ypred[i] == ytest[i]:            # Getting the count of matching labels b/w predicted and true labels
            matchCount += 1

    acc = matchCount/N * 100                # Calculating the accuracy manually

    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:
    # X = N x d
    # y = N x 1
    # Output:
    # w = d x 1

    X_T = X.transpose()
    w = np.linalg.solve(np.dot(X_T, X), np.dot(X_T, y))
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d
    # y = N x 1
    # lambd = ridge parameter (scalar)
    # Output:
    # w = d x 1

    X_T = X.transpose()
    I = np.identity(X.shape[1])
    X_T_with_reg = np.dot(X_T, X) + lambd * I
    w = np.linalg.solve(X_T_with_reg, np.dot(X_T, y))
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # rmse

    diff = ytest - np.dot(Xtest, w)
    sum = np.square(diff).sum()*(1.0 / Xtest.shape[0])
    rmse = sqrt(sum)
    return rmse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda

    # IMPLEMENT THIS METHOD
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:
    # x - a single column vector (N x 1)
    # p - integer (>= 0)
    # Outputs:
    # Xd - (N x (d+1))

    a = np.array(x).reshape(x.shape[0],1)
    b = range(p + 1)
    Xd = np.power(a,b)
    return Xd

# Main script

# Problem 1
# load the sample data
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)

plt.show()

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)

plt.show()

# Problem 2

if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('RMSE without intercept '+str(mle))
print('RMSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
plt.plot(lambdas,rmses3)

# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses4 = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    rmses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
plt.plot(lambdas,rmses4)


# Problem 5
pmax = 7
lambda_opt = lambdas[np.argmin(rmses4)]
rmses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
plt.plot(range(pmax),rmses5)
plt.legend(('No Regularization','Regularization'))
