import numpy as np
from scipy import optimize


def Sigmoid(z):
    return 1 / (1 + np.exp(-z))


def SigmoidPrime(z):
    return np.exp(-z) / ((1 + np.exp(-z))**2)


def Forward(X):
#    w1 = np.random.rand(2, 3)
#    w2 = np.random.rand(3, 1)
    w1 = np.array([[ 1.63715112,  4.05222096,  3.07920047],
       [ 1.11267933,  1.09218536,  1.73982609]])
    w2 = np.array([[ 12.70889084],
       [ 14.57323997],
       [ 14.37202815]])    
    z2 = np.dot(X, w1)
    a2 = Sigmoid(z2)
    z3 = np.dot(a2, w2)
    yhat = Sigmoid(z3)
    return yhat


def CostFunction(X, Y):
    yhat = Forward(X)
    J = 0.5*sum((Y - yhat)**2)
    return J


def CostFunctionPrime(X, Y):
    w1 = np.random.rand(2, 3)
    w2 = np.random.rand(3, 1)
    z2 = np.dot(X, w1)
    a2 = Sigmoid(z2)
    z3 = np.dot(a2, w2)
    yhat = Sigmoid(z3)
    delta3 = np.multiply((-(Y - yhat)), SigmoidPrime(z3))
    dJdW2 = np.dot(a2.T, delta3)
    delta2 = (np.dot(delta3, w2.T))*(SigmoidPrime(z2))
    dJdW1 = np.dot(X.T, delta2)
    while CostFunction(X, Y) > 0.0005:
        w1 += -0.03*dJdW1
        w2 += -0.03*dJdW2
    return w1, w2
