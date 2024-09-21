import numpy as np
import matplotlib.pyplot as plt
from PIL import Image 

def visualize(img):
    plt.imshow(img)
    plt.show()

def prepareImage(img):
    # Reshape the image into an vector Nx1

    img = img.reshape(img.shape[0] * img.shape[1] * img.shape[2], 1)

    return img

def load_input():
    # Load all images from train folder
    # Prepare X_train
    X_train = np.zeros((110, 64 * 64 * 3))
    for i in range(1, 55):
        img = plt.imread('train/cat (' + str(i) + ').jpg')
        img = img / 255.
        img = img.reshape(img.shape[0] * img.shape[1] * img.shape[2], 1)

        X_train[i, :, None] = img

    for i in range(1, 55):
        img = plt.imread('train/dog (' + str(i) + ').jpg')
        img = img / 255.
        img = img.reshape(img.shape[0] * img.shape[1] * img.shape[2], 1)

        X_train[i + 55, :, None] = img

    return X_train

def load_output():
    # Load all images from train folder
    # Prepare Y_train
    Y_train = np.zeros((110, 1))

    for i in range(55, 110):
        Y_train[i] = 1

    return Y_train

def shuffle(X_train, Y_train):
    T = np.c_[X_train, Y_train]

    np.random.shuffle(T)

    X = T[:,0:-1]
    Y = T[:,-1, None].astype(int)

    return X.T, Y.T


def init_params(n, units1, units2):
    W1 = np.random.randn(units1, n) * np.sqrt(1 / units1)
    b1 = np.random.randn(units1, 1) * np.sqrt(1 / units1)

    W2 = np.random.randn(units2, units1) * np.sqrt(1 / units2)
    b2 = np.random.randn(units2, 1) * np.sqrt(1 / units2)

    return W1, b1, W2, b2 

def ReLU(Z):
    return np.maximum(0, Z)

def deriv_ReLU(Z):
    return Z > 0

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def deriv_sigmoid(Z):
    return sigmoid(Z) * (1 - sigmoid(Z))

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)

    Z2 = W2.dot(A1) + b2
    A2 = sigmoid(Z2)

    return Z1, A1, Z2, A2

def cost_function(W1, b1, W2, b2, X, Y, m):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    J = -1 / m * np.sum(Y * np.log(np.max(A2, 1)) + (1 - Y) * np.log(np.max(1 - A2, 1)))
    return J

def model_train(X, Y, alpha, epochs):
    # Number of pixels
    n = X.shape[0]

    # Number of tests
    m = X.shape[1]

    W1, b1, W2, b2 = init_params(X.shape[0], 64, 1)

    J = []

    for i in range(epochs):
        # Forward propagation
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)

        dZ2 = A2 - Y
        dW2 = 1 / m * dZ2.dot(A1.T)
        db2 = 1 / m * np.sum(dZ2)

        dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * np.sum(dZ1)

        # Backward propagation
        W1 = W1 - alpha * dW1
        b1 = b1 - alpha * db1
        W2 = W2 - alpha * dW2
        b2 = b2 - alpha * db2

        J.append((i, cost_function(W1, b1, W2, b2, X, Y, m)))

    return J, W1, b1, W2, b2

def predict(W1, B1, W2, B2, x):
    _, _, _, A2 = forward_prop(W1, B1, W2, B2, x)
    return A2 > 0.5

def main():
    # Load all images from train folder and prepare X_train and Y_train
    X_train = load_input()
    Y_train = load_output()

    # Shuffle data
    X, Y = shuffle(X_train, Y_train)

    # Number of tests
    m = X.shape[1]

    J, W1, b1, W2, b2 = model_train(X, Y, 0.0003, 500)

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
        if predict(W1, b1, W2, b2, X[:,i, None]) == Y[0, i]:
            correct += 1
    
    print("Accuracy: ", correct / m)


if __name__ == '__main__':
    main()