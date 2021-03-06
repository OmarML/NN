import numpy as np
from scipy.optimize import minimize

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [0], [0], [1]])


class NeuralNetwork(object):
    def __init__(self, InputUnits, OutputUnits, HiddenLayers, HiddenUnits):
        # number of neurons is equal in all hidden layers for simplicity
        # still effective
        self.InputUnits = InputUnits
        self.OutputUnits = OutputUnits
        self.HiddenLayers = HiddenLayers
        self.HiddenUnits = HiddenUnits
        self.weights = None
        self.z = None

    def Logistic(self, z):
        return 1 / (1 + np.exp(-z))

    def LogisticPrime(self, z):
        return np.exp(-z) / ((1 + np.exp(-z))**2)

    def CreateWeights(self):
        w1 = np.random.rand(self.InputUnits, self.HiddenUnits)
        wl = np.random.rand(self.HiddenUnits, self.OutputUnits)
        self.weights = []
        self.weights.append(w1)
        for i in range(self.HiddenLayers - 1):
            self.weights.append(np.random.rand(self.HiddenUnits, self.HiddenUnits))
        self.weights.append(wl)
        return self.weights

    def Forward(self, X):
        new_X = self.Logistic(np.dot(X, self.weights[0]))
        for i in self.weights[1:]:
            new_X = np.dot(new_X, i)
            new_X = self.Logistic(new_X)
        return new_X

    def GetZValues(self, X):
        new_X = self.Logistic(np.dot(X, self.weights[0]))
        self.z = []
        self.z.append(np.dot(X, self.weights[0]))
        for i in self.weights[1:]:
            new_X = np.dot(new_X, i)
            self.z.append(new_X)
            new_X = self.Logistic(new_X)
        return self.z

    def GetZPrimeValues(self):
        return map(self.LogisticPrime, self.z)

    def CostFunction(self, X, Y):
        yhat = self.Forward(X)
        J = 0.5*sum((Y-yhat)**2)
        return J

    def TransposeWeights(self):
        return map(np.transpose, self.weights)

    def CalculateErrors(self, X, Y):
        delta = (-(Y - self.Forward(X)))*(self.GetZPrimeValues()[-1])
        deltas = []
        deltas.append(delta)
        t = len(self.GetZPrimeValues()) - 1
        for i in range(len(self.TransposeWeights()[1:]))[::-1]:
            delta = (np.dot(delta, self.TransposeWeights()[1:][i])) * self.GetZPrimeValues()[0:t][i]
            deltas.append(delta)
        return deltas

    def GetActivationsTransposed(self, X):
        new_X = self.Logistic(np.dot(X, self.weights[0]))
        a = [X, self.Logistic(np.dot(X, self.weights[0]))]
        for i in self.weights[1:]:
            new_X = np.dot(new_X, i)
            new_X = self.Logistic(new_X)
            a.append(new_X)
        return map(np.transpose, a)

    def WeightsDerivative(self, X, Y):
        t = len(self.GetActivationsTransposed(X)) - 1
        new_act = (self.GetActivationsTransposed(X)[0:t])[::-1]
        y = self.CalculateErrors(X, Y)
        return np.array([np.dot(new_act[i], y[i]) for i in range(len(y))])

    def GradientDescent(self, numiter, learning_rate, X, Y):
        order_deriv = self.WeightsDerivative(X, Y)[::-1]
        for i in range(numiter):
            self.weights += -learning_rate * order_deriv
        return self.Forward(X)
#            
#    def UpdateWeights(self):
#        res = minimize(self.CostFunction(X, Y), method='BFGS', jac=self.WeightsDerivative(X, Y))
a = NeuralNetwork(2, 1, 1, 3)
a.CreateWeights()
a.GetZValues(X)
a.GetZPrimeValues()
a.TransposeWeights()
a.CalculateErrors(X, Y)
a.GetActivationsTransposed(X)
a.WeightsDerivative(X, Y)
