import numpy as np
import matplotlib.pyplot as plt

# Activation functions
def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def relu(Z):
    return np.maximum(0,Z)

def drelu(Z):
    return Z > 0

def dsigmoid(Z):
    s = sigmoid(Z)
    return s*(1-s)

def softmax(Z):
    Zexp = np.exp(Z)
    sum = np.sum(Zexp, axis = 0, keepdims=True)
    return Zexp / sum

class NeuralNetwork:
    def __init__(self, layersDims, activationFunctions):
        self.layersDims = layersDims
        self.activationFunctions = activationFunctions
        self.parameters = {}
        self.L = len(layersDims)

    def initializeParameters(self):
        for l in range(1, self.L):
            self.parameters["W" + str(l)] = np.random.randn(self.layersDims[l], self.layersDims[l - 1]) * np.sqrt(2./self.layersDims[l])
            self.parameters["b" + str(l)] = np.zeros((self.layersDims[l], 1))
    
    def compile(self, learningRate, epochs, normalize = True, lambd = 1):
        self.learningRate = learningRate
        self.epochs = epochs
        self.normalize = normalize
        self.lambd = lambd

        self.initializeParameters()

    def forwardPropagation(self, X):
        cache = {}
        cache["A0"] = X
        for l in range(1, self.L):
            cache["Z" + str(l)] = np.dot(self.parameters["W" + str(l)], cache["A" + str(l-1)]) + self.parameters["b" + str(l)]

            if self.activationFunctions[l-1] == "relu":
                cache["A" + str(l)] = relu(cache["Z" + str(l)])
            elif self.activationFunctions[l-1] == "sigmoid":
                cache["A" + str(l)] = sigmoid(cache["Z" + str(l)])
            elif self.activationFunctions[l-1] == "softmax":
                cache["A" + str(l)] = softmax(cache["Z" + str(l)])
            else:
               cache["A" + str(l)] = cache["Z" + str(l)]
                
        return cache

    def backPropagation(self, cache, m):
        # Dictionary with derivatives
        dcache = {}

        for l in reversed(range(1, self.L)):
            if l == self.L - 1:
                # Check here for different loss
                AL = cache["A" + str(l)]  
                dAL = - (np.divide(self.Y, AL) - np.divide(1 - self.Y, 1 - AL))
                dcache["dA" + str(l)] = dAL
            else:
                dAl = np.dot(self.parameters["W" + str(l+1)].T, dcache["dZ" + str(l+1)])
                dcache["dA" + str(l)] = dAl 

            if self.activationFunctions[l-1] == "relu":
                dZl = dcache["dA" + str(l)] * drelu(cache["Z" + str(l)])
                dcache["dZ" + str(l)] = dZl

            if self.activationFunctions[l-1] == "sigmoid":
                dZl = dcache["dA" + str(l)] * dsigmoid(cache["Z" + str(l)])
                dcache["dZ" + str(l)] = dZl

            dWl = 1./ m * np.dot(dcache["dZ" + str(l)], cache["A" + str(l-1)].T)
            dbl = 1./ m * np.sum(dcache["dZ" + str(l)], axis=1, keepdims=True)

            dcache["dW" + str(l)] = dWl
            dcache["db" + str(l)] = dbl

        return dcache

    def updateParameters(self, dcache):
        for l in range(1, self.L):
            self.parameters["W" + str(l)] -= self.learningRate * dcache["dW" + str(l)]
            self.parameters["b" + str(l)] -= self.learningRate * dcache["db" + str(l)]

    def computeCost(self, AL):
        # Number of trainig samples
        m = self.X.shape[1]
        
        # Cost function -> Binary classification
        J = -1./m * np.sum(self.Y * np.log(np.maximum(1, AL)) + (1 - self.Y) * np.log(np.maximum(1, 1 - AL)))
        
        return J

    def fit(self, X, Y):
        self.X = X
        self.Y = Y

        # Number of trainig samples
        m = X.shape[1]

        # Function to plot J
        J = []

        # Run gradient descent
        for i in range(self.epochs):
            cache = self.forwardPropagation(X)
            dcache = self.backPropagation(cache, m)
            self.updateParameters(dcache)

            # Compute new cost
            J.append((i, self.computeCost(cache["A" + str(self.L - 1)])))
            
        return J

    def predict(self, x):
        cache = self.forwardPropagation(x)
        return cache["A" + str(self.L - 1)] > 0.5
