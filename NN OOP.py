import numpy as np


class NeuralNetwork(object):
    def __init__(self, InputUnits, OutputUnits, HiddenLayers, HiddenUnits):
        # number of neurons is equal in all hidden layers for simplicity
        # still effective
        self.InputUnits = InputUnits
        self.OutputUnits = OutputUnits
        self.HiddenLayers = HiddenLayers
        self.HiddenUnits = HiddenUnits
        self.weights = None

    def Logistic(self, z):
        return 1 / (1 + np.exp(-z))

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
            self.Logistic(new_X)
        return new_X

    def CostFunction(self, X, Y):
        yhat = self.Forward(X)
        J = 0.5*sum((Y-yhat)**2)
        return J
