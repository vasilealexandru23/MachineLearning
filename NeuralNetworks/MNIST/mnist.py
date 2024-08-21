import numpy as np
import matplotlib.pyplot as plt

def load_data():
    X = np.loadtxt(open("TrainX.csv", "rb"), delimiter=",", skiprows=0)
    Y = np.loadtxt(open("TrainY.csv", "rb"), delimiter=",", skiprows=0)

    # Combine them for shuffle
    T = np.c_[X, Y]

    # Shuffle rows
    np.random.shuffle(T)

    # Extract final data
    X = T[:,0:-1]
    Y = T[:,-1]

    # Modifie Y to one-hot encoding
    one_hot_Y = np.zeros((Y.size, int(np.max(Y)) + 1))
    one_hot_Y[np.arange(Y.size), Y.astype(int)] = 1

    return X, one_hot_Y.T

def compileModel(n, units1, units2):
    W1 = np.random.randn(units1, n) - 1.
    B1 = np.random.randn(units1, 1) - 1.

    W2 = np.random.randn(units2, units1) - 1.
    B2 = np.random.randn(units2, 1) - 1.

    return W1, B1, W2, B2

def ReLU(Z):
    return np.maximum(0, Z)

def deriv_ReLU(Z):
    return Z > 0

def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z))

def forwardPropagation(W1, B1, W2, B2, x):
    Z1 = W1.dot(x) + B1
    A1 = ReLU(Z1)

    Z2 = W2.dot(A1) + B2
    A2 = softmax(Z2)

    return A2

def cost_function(W1, B1, W2, B2, X, Y, m):
    # Compute cost function
    J = 0
    for i in range(m):
        A2 = forwardPropagation(W1, B1, W2, B2, X[i])
        # Least square error
        J += np.sum((A2 - Y[:,i]) ** 2)

    return J / (2 * m)

def fitData(W1, B1, W2, B2, X, Y, m):
    # Learning rate
    alpha = 0.1

    # Maximum iteration for gradient descent
    maxIter = 1000

    # Cost function history
    J = []

    for k in range(maxIter):
        # Forward propagation
        A0 = X.T
        Z1 = W1.dot(A0) + B1
        A1 = ReLU(Z1)
        Z2 = W2.dot(A1) + B2
        A2 = softmax(Z2)

        # Backward propagation
        dZ2 = A2 - Y
        dW2 = 1 / m * dZ2.dot(A1.T)
        dB2 = 1 / m * np.sum(dZ2)

        dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
        dW1 = 1 / m * dZ1.dot(A0.T)
        dB1 = 1 / m * np.sum(dZ1)

        # Update params
        W1 = W1 - alpha * dW1
        B1 = B1 - alpha * dB1
        W2 = W2 - alpha * dW2
        B2 = B2 - alpha * dB2

        # Compute cost function
        J.append((k, cost_function(W1, B1, W2, B2, X, Y, m)))

    return J

def predict(W1, B1, W2, B2, x):
    A2 = forwardPropagation(W1, B1, W2, B2, x)
    return np.argmax(A2)

def main():
    # Extract data
    train_x, train_y = load_data()

    train_x = train_x / 255.

    # Number of features
    n = len(train_x[0])

    # Number of trainig samples
    m = len(train_x)

    # Initialise params
    W1, B1, W2, B2 = compileModel(n, 10, 10)

    J = fitData(W1, B1, W2, B2, train_x, train_y, m)

    # Plot cost function history
    J = np.array(J)
    plt.plot(J[:,0], J[:,1])
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.show()

    print("Cost function history: ")
    for i in range(len(J)):
        print("Iteration: ", J[i][0], " Cost: ", J[i][1])

    # Get accuracy over all trainig set
    correct = 0
    for i in range(m):
        if predict(W1, B1, W2, B2, train_x[i]) == np.argmax(train_y[:,i], 0):
            correct += 1
    
    print("Accuracy: ", correct / m)

if __name__ == "__main__":
    main()

