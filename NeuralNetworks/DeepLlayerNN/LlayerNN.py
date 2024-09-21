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
            self.parameters["W" + str(l)] = np.random.randn(self.layersDims[l], self.layersDims[l - 1]) * np.sqrt(1./self.layersDims[l])
            self.parameters["b" + str(l)] = np.random.randn(self.layersDims[l], 1) * np.sqrt(1./self.layersDims[l])
    
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

    def initialize_velocity(self):
        V = {}
        for l in range(1, self.L):
            V["dW" + str(l)] = np.zeros(self.parameters["W" + str(l)].shape)
            V["db" + str(l)] = np.zeros(self.parameters["b" + str(l)].shape)
        return V

    def update_parameters_with_momentum(self, V, dcache, beta = 0.9):
        for l in range(1, self.L):
            V["dW" + str(l)] = beta * V["dW" + str(l)] + (1 - beta) * dcache["dW" + str(l)]
            V["db" + str(l)] = beta * V["db" + str(l)] + (1 - beta) * dcache["db" + str(l)]
            self.parameters["W" + str(l)] -= self.learningRate * V["dW" + str(l)]
            self.parameters["b" + str(l)] -= self.learningRate * V["db" + str(l)]

    def update_parameters_with_rmsprop(self, S, dcache, beta = 0.999, epsilon = 1e-8):
        for l in range(1, self.L):
            S["dW" + str(l)] = beta * S["dW" + str(l)] + (1 - beta) * dcache["dW" + str(l)] * dcache["dW" + str(l)]
            S["db" + str(l)] = beta * S["db" + str(l)] + (1 - beta) * dcache["db" + str(l)] * dcache["db" + str(l)]
            self.parameters["W" + str(l)] -= self.learningRate * dcache["dW" + str(l)] / (np.sqrt(S["dW" + str(l)]) + epsilon)
            self.parameters["b" + str(l)] -= self.learningRate * dcache["db" + str(l)] / (np.sqrt(S["db" + str(l)]) + epsilon)

    def update_parameters_with_adam(self, V, S, t, dcache, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
        for l in range(1, self.L):
            # Momentum
            V["dW" + str(l)] = beta1 * V["dW" + str(l)] + (1 - beta1) * dcache["dW" + str(l)]
            V["db" + str(l)] = beta1 * V["db" + str(l)] + (1 - beta1) * dcache["db" + str(l)]

            # RMSProp
            S["dW" + str(l)] = beta2 * S["dW" + str(l)] + (1 - beta2) * dcache["dW" + str(l)] * dcache["dW" + str(l)]
            S["db" + str(l)] = beta2 * S["db" + str(l)] + (1 - beta2) * dcache["db" + str(l)] * dcache["db" + str(l)]

            # Bias correction
            V_corrected_dW = V["dW" + str(l)] / (1 - beta1 ** t)
            V_corrected_db = V["db" + str(l)] / (1 - beta1 ** t)
            S_corrected_dW = S["dW" + str(l)] / (1 - beta2 ** t)
            S_corrected_db = S["db" + str(l)] / (1 - beta2 ** t)

            # Update parameters
            self.parameters["W" + str(l)] -= self.learningRate * V_corrected_dW / (np.sqrt(S_corrected_dW) + epsilon)
            self.parameters["b" + str(l)] -= self.learningRate * V_corrected_db / (np.sqrt(S_corrected_db) + epsilon)

    def mini_batch_gradient_descent(self, X, regularization = True, optimizer="None"):
        # Number of trainig samples
        m = X.shape[1]

        # Function of cost
        J = []

        # Mini-batch size
        mini_batch_size = 64

        # Get input into mini-batches of size 64
        num_mini_batches = m // mini_batch_size

        if optimizer == "momentum":
            V = self.initialize_velocity()
        elif optimizer == "rmsprop":
            S = self.initialize_velocity()
        elif optimizer == "adam":
            t = 0 # Adam counter
            V = self.initialize_velocity()
            S = self.initialize_velocity()

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

                if optimizer == "momentum":
                    self.update_parameters_with_momentum(V, dcache)
                elif optimizer == "rmsprop":
                    self.update_parameters_with_rmsprop(S, dcache)
                elif optimizer == "adam":
                    t = t + 1   # Adam counter
                    self.update_parameters_with_adam(V, S, t, dcache)
                else:
                    self.updateParameters(dcache)

                # Compute new cost
                J.append((i, self.computeCost(mini_batches_Y[j], cache["A" + str(self.L - 1)], regularization)))

        return J


    def fit(self, X, Y, regularization = True, optimizer = "None"):
        self.X = X
        self.Y = Y

        # Function to plot J
        J = self.mini_batch_gradient_descent(X, False, optimizer) 
    
        return J

    def predict(self, x):
        cache = self.forwardPropagation(x)
        return cache["A" + str(self.L - 1)] > 0.5
