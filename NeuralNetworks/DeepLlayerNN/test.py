import numpy as np
import matplotlib.pyplot as plt
from LlayerNN import NeuralNetwork

def load_input():
    # Load all images from train folder
    # Prepare X_train
    X_train = np.zeros((110, 64 * 64 * 3))
    for i in range(1, 55):
        img = plt.imread('../CatsDogs/train/cat (' + str(i) + ').jpg')
        img = img / 255.
        img = img.reshape(img.shape[0] * img.shape[1] * img.shape[2], 1)

        X_train[i, :, None] = img

    for i in range(1, 55):
        img = plt.imread('../CatsDogs/train/dog (' + str(i) + ').jpg')
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

def main():
    # Load all images from train folder and prepare X_train and Y_train
    X_train = load_input()
    Y_train = load_output()

    # Number of training samples
    m = X_train.shape[0]

    # Shuffle data
    X, Y = shuffle(X_train, Y_train)

    # Instance the model
    model = NeuralNetwork([12288, 64, 64,  1], ["relu", "relu", "sigmoid"])

    # Compile model, initialize params
    model.compile(0.01, 100)
    
    # Run gradient descent to fit the model
    J = model.fit(X, Y)

    # Plot cost function history
    J = np.array(J)
    plt.plot(J[:,0], J[:,1])
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.show()

    # Get accuracy over all trainig set
    correct = 0
    for i in range(m):
        if model.predict(X[:,i, None]) == Y[0, i]:
            correct += 1
    
    print("Accuracy: ", correct / m)

if __name__ == '__main__':
    main()