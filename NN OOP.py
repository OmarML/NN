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
    
    def CreateWeightZPairs(self):
        
        

#    def CalculateErrors(self, X, Y):
#        initial_delta = Y - self.Forward(X)
#        deltas = []
#        for i in self.HiddenLayers:
#            for i in 
        

a = NeuralNetwork(2, 1, 3, 3)
