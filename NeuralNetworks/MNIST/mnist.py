import numpy as np
import matplotlib.pyplot as plt

def load_data():
    X_train = np.loadtxt(open("TrainX.csv", "rb"), delimiter=",", skiprows=0)
    Y_train = np.loadtxt(open("TrainY.csv", "rb"), delimiter=",", skiprows=0, dtype=int)

    # Combine them for shuffle
    T = np.c_[X_train, Y_train]

    # Shuffle rows
    np.random.shuffle(T)

    # Extract final data
    X_train = T[:,0:-1]
    Y_train = T[:,-1].astype(int)

    # Modifie Y to one-hot encoding
    return X_train.T, Y_train

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def compileModel(n, units1, units2):
    W1 = np.random.randn(units1, n) * np.sqrt(1 / units1)
    B1 = np.random.randn(units1, 1) * np.sqrt(1 / units1)

    W2 = np.random.randn(units2, units1) * np.sqrt(1 / units2)
    B2 = np.random.randn(units2, 1) * np.sqrt(1 / units2)

    return W1, B1, W2, B2

def ReLU(Z):
    return np.maximum(0, Z)

def deriv_ReLU(Z):
    return Z > 0

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def cost_function(W1, B1, W2, B2, X, Y, m):
    # Compute cost function
    J = 0
    for i in range(m):
        _, _, _, A2 = forward_prop(W1, B1, W2, B2, X[:,i, None])
        J += -np.sum(Y[:,i] * np.log(A2) + (1 - Y[:,i]) * np.log(1 - A2)) 

    return J / (2 * m)

def fitData(X, Y, m, n):
    # Learning rate
    alpha = 0.9

    # Maximum iteration for gradient descent
    maxIter = 500

    # Cost function history
    J = []

    # Initialise params
    W1, B1, W2, B2 = compileModel(n, 10, 10)

    Y = one_hot(Y)

    for k in range(maxIter):
        # Forward propagation
        Z1, A1, Z2, A2 = forward_prop(W1, B1, W2, B2, X) 

        # Backward propagation
        dZ2 = A2 - Y
        dW2 = 1 / m * dZ2.dot(A1.T)
        dB2 = 1 / m * np.sum(dZ2)

        dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
        dW1 = 1 / m * dZ1.dot(X.T)
        dB1 = 1 / m * np.sum(dZ1)

        # Update params
        W1 = W1 - alpha * dW1
        B1 = B1 - alpha * dB1
        W2 = W2 - alpha * dW2
        B2 = B2 - alpha * dB2

        # Compute cost function
        J.append((k, cost_function(W1, B1, W2, B2, X, Y, m)))

    return W1, B1, W2, B2, J

def predict(W1, B1, W2, B2, x):
    _, _, _, A2 = forward_prop(W1, B1, W2, B2, x)
    return np.argmax(A2)

def main():
    # Extract data
    train_x, train_y = load_data()

    # Number of features
    n = len(train_x)

    # Number of trainig samples
    m = len(train_x[0])

    # Get all data for NN
    W1, B1, W2, B2, J = fitData(train_x, train_y, m, n)

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
        if predict(W1, B1, W2, B2, train_x[:,i, None]) == train_y[i]:
            correct += 1
    
    print("Accuracy: ", correct / m)

if __name__ == "__main__":
    main()

