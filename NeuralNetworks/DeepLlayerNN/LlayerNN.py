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
            self.parameters["b" + str(l)] = np.random.randn(self.layersDims[l], 1) * np.sqrt(2./self.layersDims[l])
    
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

    def backPropagation(self, Y, cache, m,regularization = True, epsilon = 1e-10):
        # Dictionary with derivatives
        dcache = {}

        for l in reversed(range(1, self.L)):
            if l == self.L - 1:
                # Check here for different loss
                AL = cache["A" + str(l)]  
                # Add epsilon to avoid division by zero
                # dAL = -(np.divide(self.Y, AL + epsilon) - np.divide(1 - self.Y, 1 + epsilon - AL))
                # dcache["dA" + str(l)] = dAL

                # More numeric stability to direct computation of dZ
                dZl = AL - Y
                dcache["dZ" + str(l)] = dZl
                dWl = 1./ m * np.dot(dcache["dZ" + str(l)], cache["A" + str(l-1)].T)
                dbl = 1./ m * np.sum(dcache["dZ" + str(l)], axis=1, keepdims=True)
                dcache["dW" + str(l)] = dWl
                dcache["db" + str(l)] = dbl
                continue
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

            if regularization:
                dWl += self.lambd * self.parameters["W" + str(l)]
                dbl += self.lambd * self.parameters["b" + str(l)]

            dcache["dW" + str(l)] = dWl
            dcache["db" + str(l)] = dbl

        return dcache

    def updateParameters(self, dcache):
        for l in range(1, self.L):
            self.parameters["W" + str(l)] -= self.learningRate * dcache["dW" + str(l)]
            self.parameters["b" + str(l)] -= self.learningRate * dcache["db" + str(l)]

    def computeCost(self, Y, AL, regularization, epsilon = 1e-10):
        # Number of trainig samples
        m = self.X.shape[1]
        
        # Cost function -> Binary classification
        J = -1 / m * np.sum(Y * np.log(AL + epsilon) + (1 - Y) * np.log(1 - AL + epsilon))

        if regularization:
            for l in range(1, self.L):
                J += self.lambd / (2 * m) * np.sum(np.square(self.parameters["W" + str(l)]))
        
        return J

    def batch_gradient_descent(self, X, regularization = True):
        # Number of trainig samples
        m = X.shape[1]

        # Function of cost
        J = []

        # Run gradient descent
        for i in range(self.epochs):
            cache = self.forwardPropagation(X)
            dcache = self.backPropagation(self.Y, cache, m, regularization)
            self.updateParameters(dcache)

            # Compute new cost
            J.append((i, self.computeCost(self.Y, cache["A" + str(self.L - 1)], regularization)))

        return J

    def mini_batch_gradient_descent(self, X, regularization = True):
        # Number of trainig samples
        m = X.shape[1]

        # Function of cost
        J = []

        # Mini-batch size
        mini_batch_size = 64

        # Get input into mini-batches of size 64
        num_mini_batches = m // mini_batch_size

        # Separate input into mini-batches
        mini_batches_X = []
        mini_batches_Y = []
        for i in range(num_mini_batches):
            mini_batch = X[:, i * mini_batch_size: (i + 1) * mini_batch_size]
            mini_batches_X.append(mini_batch)
            
            mini_batch = self.Y[:, i * mini_batch_size: (i + 1) * mini_batch_size]
            mini_batches_Y.append(mini_batch)

        if m % mini_batch_size != 0:
            mini_batch = X[:, num_mini_batches * mini_batch_size:]
            mini_batches_X.append(mini_batch)

            mini_batch = self.Y[:, num_mini_batches * mini_batch_size:]
            mini_batches_Y.append(mini_batch)

        # Run gradient descent
        for i in range(self.epochs):
            for j in range(num_mini_batches):
                cache = self.forwardPropagation(mini_batches_X[j])
                dcache = self.backPropagation(mini_batches_Y[j], cache, mini_batch.shape[1], regularization)
                self.updateParameters(dcache)

                # Compute new cost
                J.append((i, self.computeCost(mini_batches_Y[j], cache["A" + str(self.L - 1)], regularization)))

        return J


    def fit(self, X, Y, regularization = True):
        self.X = X
        self.Y = Y

        # Function to plot J
        J = self.batch_gradient_descent(X) 
    
        return J

    def predict(self, x):
        cache = self.forwardPropagation(x)
        return cache["A" + str(self.L - 1)] > 0.5
