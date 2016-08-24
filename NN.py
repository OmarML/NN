import numpy as np
from scipy.optimize import fmin


def Sigmoid(z):
    return 1 / (1 + np.exp(-z))


def SigmoidPrime(z):
    return np.exp(-z) / ((1 + np.exp(-z))**2)


def Forward(X):
#    w1 = np.random.rand(2, 3)
#    w2 = np.random.rand(3, 1)
    w1 = np.array([[ 1.32004664, -8.93865989,  3.74458026],
        [ 1.08999794,  3.62951108, -9.08967314]])
    w2 = np.array([[  6.71975035],
        [-11.82993103],
        [-12.03647864]])
    z2 = np.dot(X, w1)
    a2 = Sigmoid(z2)
    z3 = np.dot(a2, w2)
    yhat = Sigmoid(z3)
    return yhat


def CostFunction(X, Y):
    yhat = Forward(X)
    J = 0.5*sum((Y - yhat)**2)
    return J


def CostFunctionPrime(X, Y, numiter):
    w1 = np.random.rand(2, 3)
    w2 = np.random.rand(3, 1)
#    z2 = np.dot(X, w1)
#    a2 = Sigmoid(z2)
#    z3 = np.dot(a2, w2)
#    yhat = Sigmoid(z3)
#    delta3 = np.multiply((-(Y - yhat)), SigmoidPrime(z3))
#    dJdW2 = np.dot(a2.T, delta3)
#    delta2 = (np.dot(delta3, w2.T))*(SigmoidPrime(z2))
#    dJdW1 = np.dot(X.T, delta2)
    for i in range(0, numiter):
        z2 = np.dot(X, w1)
        a2 = Sigmoid(z2)
        z3 = np.dot(a2, w2)
        yhat = Sigmoid(z3)
        delta3 = np.multiply((-(Y - yhat)), SigmoidPrime(z3))
        dJdW2 = np.dot(a2.T, delta3)
        delta2 = (np.dot(delta3, w2.T))*(SigmoidPrime(z2))
        dJdW1 = np.dot(X.T, delta2)
        w1 += -2*dJdW1
        w2 += -2*dJdW2
    return w1, w2
